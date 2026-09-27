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
        close_code = 1002
        heartbeat_stop = asyncio.Event()
        websocket.state.send_lock = asyncio.Lock()
        try:
            await websocket.accept()
            accepted = True
            heartbeat = asyncio.create_task(_progress(websocket, heartbeat_stop))
            first = await websocket.receive()
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
            model_value = await _native_call(get_or_load_model)
            session = await _native_call(lambda: model_value.session(n_threads=2))
            decoder = await _native_call(session.stream)
            async with websocket.state.send_lock:
                await websocket.send_json({"type": "ready"})

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                if message.get("bytes") is not None:
                    pcm = _valid_pcm(message["bytes"])
                    if pcm is None:
                        await websocket.send_json({"type": "error", "message": "invalid PCM frame"})
                        return
                    await _native_call(lambda pcm=pcm: decoder.feed(pcm))
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
                await _native_call(decoder.finalize)
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
            heartbeat_stop.set()
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

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "10000")),
        ws_ping_timeout=None,
        ws_max_size=MAX_FRAME_BYTES,
    )
