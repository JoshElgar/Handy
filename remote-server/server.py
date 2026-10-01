import array
import asyncio
import hmac
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
import wave
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/parakeet-unified-en-0.6b-Q8_0.gguf")
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


async def _native_call(function):
    work = asyncio.create_task(asyncio.to_thread(function))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        try:
            await work
        finally:
            raise


def _is_cancel(text):
    try:
        return json.loads(text) == {"type": "cancel"}
    except (ValueError, TypeError):
        return False


def create_app(*, model=None, token=None):
    token = token if token is not None else _token_from_environment()
    state = {"model": model, "active": threading.Lock()}

    def get_or_load_model():
        if state["model"] is None:
            import transcribe_cpp

            state["model"] = transcribe_cpp.Model(MODEL_PATH, backend="cpu")
        return state["model"]

    @asynccontextmanager
    async def lifespan(_app):
        yield
        if state["model"] is not None:
            state["model"].close()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(request: Request):
        authorization = request.headers.get("authorization", "")
        if not hmac.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(status_code=401, detail="unauthorized")
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "audio/wav":
            raise HTTPException(status_code=415, detail="Content-Type must be audio/wav")

        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as upload:
            total = 0
            async for part in request.stream():
                total += len(part)
                if total > MAX_WAV_BYTES:
                    raise HTTPException(status_code=413, detail="WAV body exceeds 128 MiB")
                upload.write(part)
            if total == 0:
                raise HTTPException(status_code=400, detail="WAV body is empty")
            upload.seek(0)
            try:
                wav = wave.open(upload, "rb")
            except (wave.Error, EOFError, OSError, ValueError):
                raise HTTPException(status_code=400, detail="Invalid WAV file") from None
            with wav:
                channels, width, rate, frames = (
                    wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()
                )
                if channels != 1 or width != 2 or rate != 16000 or frames == 0:
                    raise HTTPException(status_code=400, detail="WAV must contain nonempty 16 kHz mono PCM16")
                if not state["active"].acquire(blocking=False):
                    return JSONResponse({"error": "transcription already in progress"}, status_code=503,
                                        headers={"Retry-After": "2"})
                session = None
                decoder = None
                try:
                    model_value = await _native_call(get_or_load_model)
                    session = await _native_call(lambda: model_value.session(n_threads=2))
                    decoder = await _native_call(session.stream)
                    remaining = frames
                    while remaining:
                        count = min(16_000, remaining)
                        raw = wav.readframes(count)
                        if len(raw) != count * 2:
                            raise HTTPException(status_code=400, detail="WAV frame data is truncated")
                        pcm16 = array.array("h")
                        pcm16.frombytes(raw)
                        if sys.byteorder == "big":
                            pcm16.byteswap()
                        pcm = array.array("f", (value / 32768.0 for value in pcm16))
                        await _native_call(lambda pcm=pcm: decoder.feed(pcm))
                        remaining -= count
                    await _native_call(decoder.finalize)
                    return {"text": decoder.text().full.strip()}
                except HTTPException:
                    raise
                except Exception:
                    LOG.exception("history transcription failed")
                    return JSONResponse({"error": "transcription failed"}, status_code=500)
                finally:
                    try:
                        if decoder is not None:
                            decoder.reset()
                    finally:
                        try:
                            if session is not None:
                                session.close()
                        finally:
                            state["active"].release()

    @app.websocket("/stream")
    async def stream(websocket: WebSocket):
        authorization = websocket.headers.get("authorization", "")
        if not hmac.compare_digest(authorization, f"Bearer {token}"):
            await websocket.close(code=1008, reason="unauthorized")
            return
        if not state["active"].acquire(blocking=False):
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": "busy"})
            await websocket.close(code=1013, reason="busy")
            return

        session = None
        decoder = None
        accepted = False
        heartbeat = None
        receiver = None
        phase = "waiting for start"
        abandoned = asyncio.Event()
        incoming = asyncio.Queue()
        queued_bytes = 0
        received_samples = 0
        processed_samples = 0
        stream_id = id(websocket)
        close_code = 1002
        heartbeat_stop = asyncio.Event()
        websocket.state.send_lock = asyncio.Lock()

        async def receive_messages():
            nonlocal queued_bytes, received_samples
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        LOG.info("stream %s disconnected during %s", stream_id, phase)
                        return
                    if message.get("text") is not None and _is_cancel(message["text"]):
                        LOG.info("stream %s cancelled during %s; waiting for active native call if any",
                                 stream_id, phase)
                        abandoned.set()
                        async with websocket.state.send_lock:
                            await websocket.send_json({"type": "cancelled"})
                            await websocket.close(code=1000)
                        return
                    data = message.get("bytes") or b""
                    queued_bytes += len(data)
                    received_samples += len(data) // 4
                    if queued_bytes > MAX_WAV_BYTES or incoming.qsize() >= 8192:
                        LOG.warning("stream %s audio backlog exceeds limit", stream_id)
                        async with websocket.state.send_lock:
                            await websocket.send_json({"type": "error", "message": "audio backlog exceeds limit"})
                            await websocket.close(code=1009)
                        return
                    incoming.put_nowait(message)
            except (WebSocketDisconnect, RuntimeError):
                LOG.info("stream %s disconnected during %s", stream_id, phase)
            except Exception:
                LOG.exception("stream %s receiver failed during %s", stream_id, phase)
            finally:
                abandoned.set()
                incoming.put_nowait({"type": "websocket.disconnect"})

        async def native(function, operation):
            nonlocal phase
            phase = operation
            started = time.monotonic()
            LOG.info("stream %s native %s started; received=%s processed=%s samples",
                     stream_id, phase, received_samples, processed_samples)
            try:
                return await _native_call(function)
            finally:
                LOG.info("stream %s native %s returned after %.3fs; abandoned=%s",
                         stream_id, operation, time.monotonic() - started, abandoned.is_set())

        try:
            await websocket.accept()
            accepted = True
            heartbeat = asyncio.create_task(_progress(websocket, heartbeat_stop))
            receiver = asyncio.create_task(receive_messages())
            first = await incoming.get()
            if abandoned.is_set():
                close_code = 1000
                return
            if first.get("text") is None:
                async with websocket.state.send_lock:
                    await websocket.send_json({"type": "error", "message": "expected start message"})
                return
            try:
                start = json.loads(first["text"])
            except (ValueError, TypeError):
                start = None
            if start != {"type": "start", "sample_rate": 16000, "format": "f32le"}:
                async with websocket.state.send_lock:
                    await websocket.send_json({"type": "error", "message": "invalid start message"})
                return
            model_value = await native(get_or_load_model, "loading model")
            if abandoned.is_set():
                return
            session = await native(lambda: model_value.session(n_threads=2), "creating session")
            if abandoned.is_set():
                return
            decoder = await native(session.stream, "creating decoder")
            if abandoned.is_set():
                return
            async with websocket.state.send_lock:
                await websocket.send_json({"type": "ready"})

            while True:
                phase = "waiting for audio"
                message = await incoming.get()
                queued_bytes -= len(message.get("bytes") or b"")
                if abandoned.is_set():
                    close_code = 1000
                    return
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    pcm = _valid_pcm(message["bytes"])
                    if pcm is None:
                        await websocket.send_json({"type": "error", "message": "invalid PCM frame"})
                        return
                    await native(lambda pcm=pcm: decoder.feed(pcm), "processing audio")
                    processed_samples += len(pcm)
                    if abandoned.is_set():
                        close_code = 1000
                        return
                    text = decoder.text()
                    async with websocket.state.send_lock:
                        await websocket.send_json({
                            "type": "partial", "committed": text.committed, "tentative": text.tentative
                        })
                    continue
                try:
                    command = json.loads(message.get("text") or "")
                except (ValueError, TypeError):
                    command = None
                if command != {"type": "finish"}:
                    await websocket.send_json({"type": "error", "message": "expected binary PCM or finish"})
                    return
                await native(decoder.finalize, "finishing")
                if abandoned.is_set():
                    close_code = 1000
                    return
                async with websocket.state.send_lock:
                    await websocket.send_json({"type": "final", "text": decoder.text().full.strip()})
                close_code = 1000
                return
        except WebSocketDisconnect:
            pass
        except Exception:
            close_code = 1011
            LOG.exception("stream transcription failed")
            if accepted:
                try:
                    async with websocket.state.send_lock:
                        await websocket.send_json({"type": "error", "message": "transcription failed"})
                except Exception:
                    pass
        finally:
            LOG.info("stream %s cleanup started during %s", stream_id, phase)
            heartbeat_stop.set()
            if receiver:
                receiver.cancel()
                try:
                    await receiver
                except asyncio.CancelledError:
                    pass
            try:
                if heartbeat:
                    heartbeat.cancel()
                    try:
                        await heartbeat
                    except asyncio.CancelledError:
                        pass
                if accepted:
                    try:
                        await websocket.close(code=close_code)
                    except Exception:
                        pass
            finally:
                try:
                    if decoder is not None:
                        decoder.reset()
                finally:
                    try:
                        if session is not None:
                            session.close()
                    finally:
                        state["active"].release()
                        LOG.info("stream %s slot released; received=%s processed=%s samples",
                                 stream_id, received_samples, processed_samples)

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
