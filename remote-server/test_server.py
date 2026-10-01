import array
import io
import os
import struct
import threading
import sys
import time
import types
import unittest
import wave
from unittest.mock import patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

os.environ.setdefault("API_TOKEN", "test-secret")
sys.modules.setdefault("transcribe_cpp", types.SimpleNamespace())
from server import create_app


def wav_bytes(*, seconds=0.01, rate=16000, channels=1, width=2):
    frames = int(seconds * rate)
    pcm = array.array("h", [100] * frames * channels).tobytes()
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return out.getvalue()


class FakeStream:
    def __init__(self, owner):
        self.owner = owner

    def feed(self, pcm):
        if self.owner.fail_feed:
            raise RuntimeError("native feed failure")
        self.owner.feed_started.set()
        self.owner.feed_release.wait()
        if self.owner.feed_delay:
            time.sleep(self.owner.feed_delay)
        self.owner.received.extend(pcm)
        return type("Update", (), {"committed_changed": True, "tentative_changed": True})()

    def text(self):
        received = len(self.owner.received)
        return type("Text", (), {
            "full": f"heard {received} now",
            "committed": f"heard {received}",
            "tentative": " now",
        })()

    def finalize(self):
        if self.owner.finalize_delay:
            time.sleep(self.owner.finalize_delay)
        self.owner.finalized_samples = len(self.owner.received)
        return None

    def reset(self):
        self.owner.reset_count += 1


class FakeSession:
    def __init__(self, owner):
        self.owner = owner

    def stream(self, **_kwargs):
        return FakeStream(self.owner)

    def close(self):
        self.owner.session_close_count += 1


class FakeModel:
    def __init__(self):
        self.received = []
        self.finalized_samples = None
        self.transcribed_samples = 0
        self.reset_count = 0
        self.session_close_count = 0
        self.feed_started = threading.Event()
        self.feed_release = threading.Event()
        self.feed_release.set()
        self.feed_delay = 0
        self.finalize_delay = 0
        self.fail_feed = False
        self.fail_run = False

    def session(self, **_kwargs):
        return FakeSession(self)

    def close(self):
        pass


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.model = FakeModel()
        self.app = create_app(model=self.model, token="test-secret")
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def connect(self, *, token="test-secret"):
        return self.client.websocket_connect(
            "/stream", headers={"Authorization": f"Bearer {token}"}
        )

    def start(self, ws):
        ws.send_json({"type": "start", "sample_rate": 16000, "format": "f32le"})
        self.assertEqual(ws.receive_json(), {"type": "ready"})

    def wait_for_resets(self, expected):
        for _ in range(100):
            if self.model.reset_count >= expected:
                return
            time.sleep(0.01)
        self.fail(f"expected {expected} session resets, got {self.model.reset_count}")

    def wait_for_close_count(self, expected):
        for _ in range(100):
            if self.model.session_close_count >= expected:
                return
            time.sleep(0.01)
        self.fail(f"expected {expected} session closes, got {self.model.session_close_count}")

    def test_health_needs_no_token(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})

    def test_rejects_websocket_without_bearer_token(self):
        with self.assertRaises(WebSocketDisconnect) as raised:
            with self.connect(token="wrong"):
                pass
        self.assertEqual(raised.exception.code, 1008)

    def test_stream_preserves_order_and_drains_every_frame_before_finish(self):
        with self.connect() as ws:
            self.start(ws)
            chunks = [struct.pack("<4f", 0.1, 0.2, 0.3, 0.4), struct.pack("<2f", -0.5, 0.6)]
            for chunk in chunks:
                ws.send_bytes(chunk)
            ws.send_json({"type": "finish"})
            partials = []
            while True:
                message = ws.receive_json()
                if message["type"] == "final":
                    self.assertEqual(message, {"type": "final", "text": "heard 6 now"})
                    break
                partials.append(message)
        self.assertEqual([item["type"] for item in partials], ["partial", "partial"])
        self.assertEqual(partials[-1]["committed"], "heard 6")
        self.wait_for_resets(1)
        self.assertEqual(len(self.model.received), 6)
        for actual, expected in zip(self.model.received, [0.1, 0.2, 0.3, 0.4, -0.5, 0.6]):
            self.assertAlmostEqual(actual, expected, places=6)
        self.assertEqual(self.model.finalized_samples, 6)
        self.assertEqual(self.model.reset_count, 1)
        self.assertEqual(self.model.session_close_count, 1)

    def test_progress_continues_during_slow_native_finalize(self):
        self.model.finalize_delay = 0.08
        with patch("server.PROGRESS_INTERVAL", 0.01):
            with self.connect() as ws:
                self.start(ws)
                ws.send_json({"type": "finish"})
                seen = []
                while not any(message["type"] == "final" for message in seen):
                    seen.append(ws.receive_json())
        self.assertTrue(any(message == {"type": "progress"} for message in seen))

    def test_rejects_malformed_or_oversized_audio_frame(self):
        for chunk in (b"\x00", struct.pack("<f", float("nan")), b"\x00" * 65540):
            with self.subTest(size=len(chunk)):
                closed = self.model.session_close_count
                with self.connect() as ws:
                    self.start(ws)
                    ws.send_bytes(chunk)
                    self.assertEqual(ws.receive_json()["type"], "error")
                self.wait_for_close_count(closed + 1)

    def test_busy_connection_does_not_block_health(self):
        with self.connect() as active:
            self.start(active)
            with self.connect() as busy:
                self.assertEqual(busy.receive_json(), {"type": "error", "message": "busy"})
                with self.assertRaises(Exception):
                    busy.receive_json()
            self.assertEqual(self.client.get("/health").status_code, 200)

    def test_feed_failure_releases_model_for_next_stream(self):
        self.model.fail_feed = True
        with self.connect() as ws:
            self.start(ws)
            ws.send_bytes(struct.pack("<f", 0.25))
            self.assertEqual(ws.receive_json()["type"], "error")
        self.wait_for_resets(1)
        self.model.fail_feed = False
        with self.connect() as ws:
            self.start(ws)
            ws.send_json({"type": "finish"})
            self.assertEqual(ws.receive_json()["type"], "final")

    def test_disconnect_resets_session_and_allows_reuse(self):
        with self.connect() as ws:
            self.start(ws)
            ws.send_bytes(struct.pack("<f", 0.25))
            ws.receive_json()
        self.wait_for_resets(1)
        with self.connect() as ws:
            self.start(ws)
            ws.send_json({"type": "finish"})
            self.assertEqual(ws.receive_json()["type"], "final")

    def test_cancel_discards_backlog_and_releases_slot_after_active_feed(self):
        self.model.feed_release.clear()
        try:
            with self.connect() as ws:
                self.start(ws)
                ws.send_bytes(struct.pack("<f", 0.25))
                self.assertTrue(self.model.feed_started.wait(1))
                for _ in range(10):
                    ws.send_bytes(struct.pack("<f", 0.5))
                with self.assertLogs("parakeet-server", level="INFO") as logs:
                    ws.send_json({"type": "cancel"})
                    # The receiver must notice cancel while the native call is blocked.
                    self.assertEqual(ws.receive_json(), {"type": "cancelled"})
                self.assertTrue(any("processing" in line for line in logs.output))
                with self.connect() as busy:
                    self.assertEqual(busy.receive_json(), {"type": "error", "message": "busy"})
                self.model.feed_release.set()
            self.wait_for_close_count(1)
            self.assertEqual(len(self.model.received), 1)
            self.assertIsNone(self.model.finalized_samples)
            with self.connect() as next_ws:
                self.start(next_ws)
                next_ws.send_json({"type": "finish"})
                self.assertEqual(next_ws.receive_json()["type"], "final")
        finally:
            self.model.feed_release.set()

    def test_cancel_before_ready_does_not_load_model(self):
        with self.connect() as ws:
            ws.send_json({"type": "cancel"})
            self.assertEqual(ws.receive_json(), {"type": "cancelled"})
        self.wait_for_close_count(0)
        with self.connect() as ws:
            self.start(ws)
            ws.send_json({"type": "cancel"})
            self.assertEqual(ws.receive_json(), {"type": "cancelled"})
        self.wait_for_close_count(1)

    def test_history_retry_accepts_audio_longer_than_one_minute(self):
        response = self.client.post(
            "/transcribe",
            content=wav_bytes(seconds=61),
            headers={"Authorization": "Bearer test-secret", "Content-Type": "audio/wav"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": f"heard {16_000 * 61} now"})
        self.assertEqual(self.model.finalized_samples, 16_000 * 61)

    def test_history_retry_auth_and_wav_validation(self):
        base = {"Content-Type": "audio/wav"}
        self.assertEqual(self.client.post("/transcribe", content=wav_bytes(), headers=base).status_code, 401)
        headers = {**base, "Authorization": "Bearer test-secret"}
        self.assertEqual(self.client.post("/transcribe", content=b"bad", headers=headers).status_code, 400)
        self.assertEqual(
            self.client.post("/transcribe", content=wav_bytes(rate=8000), headers=headers).status_code,
            400,
        )


if __name__ == "__main__":
    unittest.main()
