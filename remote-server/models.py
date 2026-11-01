"""Handy's pinned catalogue and verified on-demand downloads."""
import hashlib
import json
import os
from pathlib import Path
import urllib.request

DEFAULT_MODEL = "parakeet-unified-en-0.6b"


class ModelRegistry:
    def __init__(self, *, data=None, cache_dir=None):
        if data is None:
            path = Path(__file__).with_name("catalog.json")
            if not path.exists():
                path = Path(__file__).parent.parent / "src-tauri/src/catalog/catalog.json"
            data = json.loads(path.read_text())
        self.cache_dir = Path(cache_dir or os.environ.get("MODEL_CACHE_DIR", "/app/models"))
        self.models = {}
        for model in data["models"]:
            file = next(f for f in model["files"] if f["quant"] == model["default_quant"])
            self.models[model["slug"]] = {**model, "file": file}

    def get(self, model_id):
        if model_id is None:
            model_id = DEFAULT_MODEL
        if not isinstance(model_id, str) or model_id not in self.models:
            raise ValueError("Unknown remote transcription model")
        return self.models[model_id]

    def list(self):
        return [{"id": m["slug"], "name": m["name"], "languages": m["languages"],
                 "supports_streaming": m["capabilities"]["streaming"],
                 "supports_language_detection": m["capabilities"]["lang_detect"],
                 "size_bytes": m["file"]["size_bytes"]} for m in self.models.values()]

    @staticmethod
    def _verified(path, file):
        if not path.is_file() or path.stat().st_size != file["size_bytes"]:
            return False
        with path.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        return digest == file["sha256"]

    def ensure_file(self, model_id):
        model = self.get(model_id)
        file = model["file"]
        directory = self.cache_dir / model["slug"] / model["revision"]
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / file["filename"]
        if self._verified(path, file):
            return path
        path.unlink(missing_ok=True)
        temporary = path.with_suffix(".part")
        url = f"https://huggingface.co/{model['id']}/resolve/{model['revision']}/{file['filename']}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > file["size_bytes"]:
                        raise ValueError("Model download exceeds catalogue size")
                    output.write(chunk)
            if not self._verified(temporary, file):
                raise ValueError("Model download checksum does not match catalogue")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path
