import array
import asyncio
import hmac
import json
import logging
import math
import os
import sys
import tempfile
import base64
from pathlib import Path
import time
import wave
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/app/models/parakeet-unified-en-0.6b-Q8_0.gguf"
)
MAX_FRAME_BYTES = 64 * 1024
MAX_WAV_BYTES = 128 * 1024 * 1024
PROGRESS_INTERVAL = 10
LOG = logging.getLogger("parakeet-server")


def _token_from_environment():
    token = os.environ.get("API_TOKEN", "")
    token_file = os.environ.get("API_TOKEN_FILE")
    if token_file:
        try:
            with open(token_file, encoding="utf-8") as secret:
                file_token = secret.read().strip()
        except OSError as error:
            raise RuntimeError("API_TOKEN_FILE could not be read") from error
        if token and not hmac.compare_digest(token, file_token):
            raise RuntimeError("API_TOKEN and API_TOKEN_FILE do not match")
        token = token or file_token
    if not token:
        raise RuntimeError("Set API_TOKEN or API_TOKEN_FILE")
    return token


def _valid_pcm(data):
    if not data or len(data) > MAX_FRAME_BYTES or len(data) % 4:
        return None
    samples = array.array("f")
    samples.frombytes(data)
    if sys.byteorder == "big":
        samples.byteswap()
    if not all(math.isfinite(sample) for sample in samples):
        return None
    return samples


async def _progress(websocket, stop):
    send_lock = websocket.state.send_lock
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=PROGRESS_INTERVAL)
        except asyncio.TimeoutError:
            try:
                async with send_lock:
                    await websocket.send_json({"type": "progress"})
            except Exception:
                return


def create_app(*, token=None, worker_command=None):
    token = token if token is not None else _token_from_environment()
    worker_command = worker_command or [
        sys.executable,
        "-u",
        str(Path(__file__).with_name("model_worker.py")),
    ]
    state = {"worker": None, "active": None}
    lock = asyncio.Lock()

    async def kill_worker():
        worker = state["worker"]
        state["worker"] = None
        if worker is not None:
            if worker.returncode is None:
                try:
                    worker.kill()
                except ProcessLookupError:
                    pass
            await worker.communicate()
            LOG.info("model process %s stopped", worker.pid)

    async def claim():
        owner = {"task": asyncio.current_task(), "replaced": False}
        async with lock:
            old = state["active"]
            if old is not None:
                old["replaced"] = True
                old["task"].cancel()
                await kill_worker()
            state["active"] = owner
        return owner

    async def release(owner, completed):
        async with lock:
            # Old cleanup must never stop the replacement's model process.
            if state["active"] is owner:
                if not completed:
                    await kill_worker()
                state["active"] = None

    async def call(owner, command):
        if state["active"] is not owner:
            raise asyncio.CancelledError()
        worker = state["worker"]
        if worker is None:
            spawning = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *worker_command,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    limit=4 * 1024 * 1024,
                    env={**os.environ, "MODEL_PATH": MODEL_PATH},
                )
            )
            try:
                worker = await asyncio.shield(spawning)
            except asyncio.CancelledError:
                worker = await spawning
                try:
                    worker.kill()
                except ProcessLookupError:
                    pass
                await worker.communicate()
                raise
            state["worker"] = worker
            LOG.info("model process %s started", worker.pid)
        started = time.monotonic()
        LOG.info("model %s %s started", worker.pid, command["type"])
        worker.stdin.write((json.dumps(command) + "\n").encode())
        await worker.stdin.drain()
        line = await worker.stdout.readline()
        if not line:
            raise RuntimeError("Model process exited before replying")
        LOG.info(
            "model %s %s returned after %.3fs",
            worker.pid,
            command["type"],
            time.monotonic() - started,
        )
        return json.loads(line)

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            await kill_worker()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(request: Request):
        if not hmac.compare_digest(
            request.headers.get("authorization", ""), f"Bearer {token}"
        ):
            raise HTTPException(status_code=401, detail="unauthorized")
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "audio/wav"
        ):
            raise HTTPException(
                status_code=415, detail="Content-Type must be audio/wav"
            )
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as upload:
            total = 0
            async for part in request.stream():
                total += len(part)
                if total > MAX_WAV_BYTES:
                    raise HTTPException(
                        status_code=413, detail="WAV body exceeds 128 MiB"
                    )
                upload.write(part)
            upload.seek(0)
            try:
                wav = wave.open(upload, "rb")
            except (wave.Error, EOFError, OSError, ValueError):
                raise HTTPException(
                    status_code=400, detail="Invalid WAV file"
                ) from None
            with wav:
                frames = wav.getnframes()
                if (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) != (
                    1,
                    2,
                    16000,
                ) or not frames:
                    raise HTTPException(
                        status_code=400,
                        detail="WAV must contain nonempty 16 kHz mono PCM16",
                    )
                owner = await claim()
                completed = False
                try:
                    await call(owner, {"type": "start"})
                    remaining = frames
                    while remaining:
                        count = min(16000, remaining)
                        raw = wav.readframes(count)
                        if len(raw) != count * 2:
                            raise HTTPException(
                                status_code=400, detail="WAV frame data is truncated"
                            )
                        pcm16 = array.array("h")
                        pcm16.frombytes(raw)
                        if sys.byteorder == "big":
                            pcm16.byteswap()
                        pcm = array.array("f", (value / 32768.0 for value in pcm16))
                        if sys.byteorder == "big":
                            pcm.byteswap()
                        await call(
                            owner,
                            {
                                "type": "feed",
                                "audio": base64.b64encode(pcm.tobytes()).decode(),
                            },
                        )
                        remaining -= count
                    result = await call(owner, {"type": "finish"})
                    completed = True
                    await release(owner, True)
                    return {"text": result["text"]}
                except asyncio.CancelledError:
                    if not owner["replaced"]:
                        raise
                    return JSONResponse(
                        {"error": "replaced by a newer recording"}, status_code=409
                    )
                except HTTPException:
                    raise
                except Exception:
                    LOG.exception("history transcription failed")
                    return JSONResponse(
                        {"error": "transcription failed"}, status_code=500
                    )
                finally:
                    await release(owner, completed)

    @app.websocket("/stream")
    async def stream(websocket: WebSocket):
        if not hmac.compare_digest(
            websocket.headers.get("authorization", ""), f"Bearer {token}"
        ):
            await websocket.close(code=1008, reason="unauthorized")
            return
        await websocket.accept()
        websocket.state.send_lock = asyncio.Lock()
        owner = None
        receiver = heartbeat = None
        completed = False
        cancelled = disconnected = False
        incoming = asyncio.Queue()
        queued_bytes = 0
        stop = asyncio.Event()
        task = asyncio.current_task()

        async def send(message):
            async with websocket.state.send_lock:
                await websocket.send_json(message)

        async def receive():
            nonlocal cancelled, disconnected, queued_bytes
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        disconnected = True
                        return
                    if message.get("text") is not None:
                        try:
                            command = json.loads(message["text"])
                        except (ValueError, TypeError):
                            command = None
                        if command == {"type": "cancel"}:
                            cancelled = True
                            return
                    queued_bytes += len(message.get("bytes") or b"")
                    if queued_bytes > MAX_WAV_BYTES or incoming.qsize() >= 8192:
                        await send(
                            {"type": "error", "message": "audio backlog exceeds limit"}
                        )
                        return
                    incoming.put_nowait(message)
            except (WebSocketDisconnect, RuntimeError):
                disconnected = True
            finally:
                if not completed:
                    task.cancel()

        try:
            first = await asyncio.wait_for(websocket.receive(), timeout=20)
            try:
                start = json.loads(first.get("text") or "")
            except (ValueError, TypeError):
                start = None
            if start == {"type": "cancel"}:
                await send({"type": "cancelled"})
                return
            if start != {"type": "start", "sample_rate": 16000, "format": "f32le"}:
                await send({"type": "error", "message": "invalid start message"})
                return
            owner = await claim()
            receiver = asyncio.create_task(receive())
            heartbeat = asyncio.create_task(_progress(websocket, stop))
            await send(await call(owner, {"type": "start"}))
            while True:
                message = await incoming.get()
                audio = message.get("bytes")
                if audio is not None:
                    queued_bytes -= len(audio)
                    if _valid_pcm(audio) is None:
                        await send({"type": "error", "message": "invalid PCM frame"})
                        return
                    await send(
                        await call(
                            owner,
                            {"type": "feed", "audio": base64.b64encode(audio).decode()},
                        )
                    )
                    continue
                try:
                    command = json.loads(message.get("text") or "")
                except (ValueError, TypeError):
                    command = None
                if command != {"type": "finish"}:
                    await send(
                        {"type": "error", "message": "expected binary PCM or finish"}
                    )
                    return
                result = await call(owner, {"type": "finish"})
                completed = True
                await release(owner, True)
                await send(result)
                return
        except asyncio.CancelledError:
            if owner is not None:
                await release(owner, False)
            try:
                if owner is not None and owner["replaced"]:
                    await send(
                        {"type": "error", "message": "replaced by a newer recording"}
                    )
                elif cancelled:
                    await send({"type": "cancelled"})
                elif not disconnected:
                    raise
            except (WebSocketDisconnect, RuntimeError):
                pass
        except WebSocketDisconnect:
            pass
        except Exception:
            LOG.exception("stream transcription failed")
            try:
                await send({"type": "error", "message": "transcription failed"})
            except Exception:
                pass
        finally:
            # Prevent receiver shutdown from cancelling this cleanup.
            stop.set()
            finished = completed
            completed = True
            for background in (receiver, heartbeat):
                if background:
                    background.cancel()
                    try:
                        await background
                    except asyncio.CancelledError:
                        pass
            if owner is not None:
                await release(owner, finished)
            try:
                await websocket.close(code=1000)
            except Exception:
                pass

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        workers=1,
        ws_ping_interval=20,
        ws_ping_timeout=20,
        ws_max_size=MAX_FRAME_BYTES,
    )
