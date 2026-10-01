import base64
import io
import os
import struct
import sys
import unittest
import wave
from pathlib import Path

from fastapi.testclient import TestClient

os.environ.setdefault("API_TOKEN", "test-secret")
from server import create_app

# Spawn this same file as a deliberately small fake native model.
if __name__ == "__main__" and "--worker" in sys.argv:
    samples = 0
    for line in sys.stdin:
        command = __import__("json").loads(line)
        kind = command["type"]
        if kind == "start":
            samples = 0
            result = {"type": "ready", "pid": os.getpid()}
        elif kind == "feed":
            audio = base64.b64decode(command["audio"])
            value = struct.unpack("<f", audio[:4])[0]
            if value == 0.75:
                print(__import__("json").dumps({"type": "working"}), flush=True)
                while True:
                    __import__("time").sleep(10)
            if value == -0.75:
                os._exit(1)
            samples += len(audio) // 4
            result = {
                "type": "partial",
                "committed": f"heard {samples}",
                "tentative": "",
            }
        elif kind == "finish":
            result = {"type": "final", "text": f"heard {samples}"}
        else:
            result = {"type": "done"}
        print(__import__("json").dumps(result), flush=True)
    sys.exit(0)


def wav_bytes():
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(struct.pack("<h", 100) * 160)
    return output.getvalue()


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(
            token="test-secret",
            worker_command=[
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
            ],
        )
        self.client = TestClient(self.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def connect(self, token="test-secret"):
        return self.client.websocket_connect(
            "/stream", headers={"Authorization": f"Bearer {token}"}
        )

    def start(self, ws):
        ws.send_json({"type": "start", "sample_rate": 16000, "format": "f32le"})
        message = ws.receive_json()
        self.assertEqual(message["type"], "ready")
        return message["pid"]

    def finish(self, ws):
        ws.send_json({"type": "finish"})
        self.assertEqual(ws.receive_json()["type"], "final")

    def test_new_recording_kills_hung_model_and_reuses_successful_worker(self):
        with self.connect() as old:
            old_pid = self.start(old)
            old.send_bytes(struct.pack("<f", 0.75))
            self.assertEqual(old.receive_json()["type"], "working")
            with self.connect() as newest:
                new_pid = self.start(newest)
                self.assertNotEqual(old_pid, new_pid)
                with self.assertRaises(ProcessLookupError):
                    os.kill(old_pid, 0)
                self.assertEqual(
                    old.receive_json(),
                    {"type": "error", "message": "replaced by a newer recording"},
                )
                newest.send_bytes(struct.pack("<f", 0.25))
                self.assertEqual(newest.receive_json()["committed"], "heard 1")
                self.finish(newest)
            with self.connect() as next_ws:
                self.assertEqual(self.start(next_ws), new_pid)
                self.finish(next_ws)

    def test_cancel_kills_hung_call_and_accepts_next_recording(self):
        with self.connect() as ws:
            pid = self.start(ws)
            ws.send_bytes(struct.pack("<f", 0.75))
            self.assertEqual(ws.receive_json()["type"], "working")
            ws.send_json({"type": "cancel"})
            self.assertEqual(ws.receive_json(), {"type": "cancelled"})
        with self.connect() as next_ws:
            self.assertNotEqual(self.start(next_ws), pid)
            self.finish(next_ws)

    def test_disconnect_during_hung_call_allows_next_recording(self):
        with self.connect() as ws:
            pid = self.start(ws)
            ws.send_bytes(struct.pack("<f", 0.75))
            ws.receive_json()
        with self.connect() as next_ws:
            self.assertNotEqual(self.start(next_ws), pid)
            self.finish(next_ws)

    def test_finish_drains_audio_in_order(self):
        with self.connect() as ws:
            self.start(ws)
            ws.send_bytes(struct.pack("<2f", 0.1, 0.2))
            ws.send_bytes(struct.pack("<f", 0.3))
            ws.send_json({"type": "finish"})
            self.assertEqual(ws.receive_json()["committed"], "heard 2")
            self.assertEqual(ws.receive_json()["committed"], "heard 3")
            self.assertEqual(ws.receive_json(), {"type": "final", "text": "heard 3"})

    def test_worker_crash_allows_next_recording(self):
        with self.connect() as ws:
            self.start(ws)
            ws.send_bytes(struct.pack("<f", -0.75))
            self.assertEqual(ws.receive_json()["type"], "error")
        with self.connect() as ws:
            self.start(ws)
            self.finish(ws)

    def test_invalid_or_unauthorized_start_does_not_replace_active_recording(self):
        with self.connect() as active:
            self.start(active)
            with self.assertRaises(Exception):
                with self.connect("wrong"):
                    pass
            with self.connect() as invalid:
                invalid.send_json({"type": "invalid"})
                self.assertEqual(invalid.receive_json()["type"], "error")
            self.finish(active)
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_audio_validation_and_history_retry(self):
        for audio in (b"x", struct.pack("<f", float("nan")), bytes(65540)):
            with self.connect() as ws:
                self.start(ws)
                ws.send_bytes(audio)
                self.assertEqual(ws.receive_json()["type"], "error")
        headers = {"Authorization": "Bearer test-secret", "Content-Type": "audio/wav"}
        response = self.client.post("/transcribe", content=wav_bytes(), headers=headers)
        self.assertEqual(response.json(), {"text": "heard 160"})
        self.assertEqual(
            self.client.post(
                "/transcribe", content=b"bad", headers=headers
            ).status_code,
            400,
        )
        self.assertEqual(
            self.client.post("/transcribe", content=wav_bytes()).status_code, 401
        )


if __name__ == "__main__":
    unittest.main()
