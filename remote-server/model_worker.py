import array
import base64
import json
import os
import sys

from models import ModelRegistry


class ModelWorker:
    def __init__(self, engine, registry):
        self.engine = engine
        self.registry = registry
        self.model = self.session = self.decoder = None
        self.model_id = None
        self.pcm = array.array("f")
        self.received = 0

    def close_session(self):
        if self.decoder is not None:
            self.decoder.reset()
            self.decoder = None
        if self.session is not None:
            self.session.close()
            self.session = None
        self.pcm = array.array("f")

    def close(self):
        self.close_session()
        if self.model is not None:
            self.model.close()
            self.model = None
            self.model_id = None

    def handle(self, command):
        kind = command["type"]
        if kind == "start":
            entry = self.registry.get(command.get("model"))
            self.close_session()
            if entry["slug"] != self.model_id:
                self.close()
                path = self.registry.ensure_file(entry["slug"])
                self.model = self.engine.Model(path, backend="cpu")
                self.model_id = entry["slug"]
            self.caps = self.model.capabilities
            self.session = self.model.session(n_threads=2)
            languages = self.caps.languages or tuple(entry["languages"])
            intent = command.get("language", "auto")
            if intent in ("zh-Hans", "zh-Hant"):
                intent = "zh"
            self.language = intent if intent in languages else (
                None if self.caps.supports_language_detect else (
                    "en" if "en" in languages else languages[0] if languages else None))
            self.received = 0
            self.decoder = self.session.stream(language=self.language) if self.caps.supports_streaming else None
            return {"type": "ready", "model": self.model_id, "streaming": self.decoder is not None}
        if self.session is None:
            raise ValueError("Model session has not started")
        if kind == "feed":
            pcm = array.array("f")
            pcm.frombytes(base64.b64decode(command["audio"], validate=True))
            if sys.byteorder == "big":
                pcm.byteswap()
            self.received += len(pcm)
            if self.caps.max_audio_ms and self.received * 1000 > self.caps.max_audio_ms * 16000:
                raise ValueError(f"Recording exceeds this model's {self.caps.max_audio_ms / 1000:g}s input limit")
            if self.decoder is None:
                self.pcm.extend(pcm)
                return {"type": "partial", "committed": "", "tentative": ""}
            self.decoder.feed(pcm)
            text = self.decoder.text()
            return {"type": "partial", "committed": text.committed, "tentative": text.tentative}
        if kind == "finish":
            if self.decoder is not None:
                self.decoder.finalize()
                result = self.decoder.snapshot()
                text = self.decoder.text().full.strip()
            elif self.pcm:
                result = self.session.run(self.pcm, language=self.language, timestamps="none")
                text = result.text.strip()
            else:
                result = None
                text = ""
            reply = {"type": "final", "text": text, "language": result.language if result else self.language}
            self.close_session()
            return reply
        raise ValueError("Invalid worker command")


def run():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import transcribe_cpp

    worker = ModelWorker(transcribe_cpp, ModelRegistry())
    try:
        for line in sys.stdin:
            try:
                reply = worker.handle(json.loads(line))
            except Exception as error:
                worker.close()
                print(f"Model operation failed: {error}", file=sys.stderr, flush=True)
                reply = {"type": "error", "message": f"Model operation failed: {error}"}
            protocol.write(json.dumps(reply) + "\n")
    finally:
        worker.close()


if __name__ == "__main__":
    run()
