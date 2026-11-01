import array
import hashlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models import ModelRegistry
from model_worker import ModelWorker


def catalog():
    def entry(slug, streaming):
        return {
            "id": f"handy-computer/{slug}-gguf", "slug": slug,
            "revision": "a" * 40, "name": slug, "languages": ["en"],
            "capabilities": {"streaming": streaming, "translate": False, "lang_detect": False},
            "default_quant": "Q8_0", "files": [{"filename": f"{slug}-Q8_0.gguf",
                "quant": "Q8_0", "size_bytes": 4, "sha256": hashlib.sha256(b"gguf").hexdigest()}],
        }
    return {"models": [entry("parakeet-unified-en-0.6b", True), entry("batch", False)]}


class RegistryTests(unittest.TestCase):
    def test_all_catalog_models_have_unique_downloadable_defaults(self):
        registry = ModelRegistry()
        self.assertEqual(len(registry.list()), 69)
        self.assertEqual(len({m["id"] for m in registry.list()}), 69)
        for model in registry.list():
            self.assertTrue(registry.get(model["id"])["file"]["sha256"])
        with self.assertRaises(ValueError):
            registry.get("../../foreign.gguf")

    def test_download_is_verified_cached_and_corrupt_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = ModelRegistry(data=catalog(), cache_dir=directory)
            with patch("models.urllib.request.urlopen", return_value=io.BytesIO(b"gguf")) as download:
                path = registry.ensure_file(None)
                self.assertEqual(path.read_bytes(), b"gguf")
                self.assertEqual(registry.ensure_file(None), path)
                self.assertEqual(download.call_count, 1)
            path.write_bytes(b"oops")
            with patch("models.urllib.request.urlopen", return_value=io.BytesIO(b"oops")):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    registry.ensure_file(None)
            self.assertFalse(path.exists())
            self.assertFalse(list(Path(directory).rglob("*.part")))


class FakeEngine:
    def __init__(self):
        self.live = 0
        self.loads = []
        engine = self
        class Model:
            def __init__(self, path, backend):
                engine.live += 1
                if engine.live > 1:
                    raise AssertionError("two models loaded")
                engine.loads.append(path)
                self.streaming = "batch" not in str(path)
            @property
            def capabilities(self):
                return SimpleNamespace(supports_streaming=self.streaming, languages=("en",),
                    supports_language_detect=False, supports_translate=False, max_audio_ms=100)
            def session(self, n_threads):
                return Session()
            def close(self):
                engine.live -= 1
        class Session:
            def stream(self, **kwargs):
                return Decoder()
            def run(self, pcm, **kwargs):
                return SimpleNamespace(text=f"batch {len(pcm)}", language="en")
            def close(self):
                pass
        class Decoder:
            def feed(self, pcm):
                pass
            def text(self):
                return SimpleNamespace(committed="live", tentative="", full="live")
            def snapshot(self):
                return SimpleNamespace(language="en")
            def finalize(self):
                pass
            def reset(self):
                pass
        self.Model = Model


class WorkerTests(unittest.TestCase):
    def test_stream_batch_reuse_and_switch_keep_one_model_loaded(self):
        engine = FakeEngine()
        registry = ModelRegistry(data=catalog())
        with patch.object(registry, "ensure_file", side_effect=lambda mid: Path(registry.get(mid)["slug"])):
            worker = ModelWorker(engine, registry)
            self.assertTrue(worker.handle({"type": "start"})["streaming"])
            worker.handle({"type": "feed", "audio": "AAAAAA=="})
            self.assertEqual(worker.handle({"type": "finish"})["text"], "live")
            worker.handle({"type": "start"})
            worker.handle({"type": "finish"})
            self.assertEqual(len(engine.loads), 1)
            self.assertFalse(worker.handle({"type": "start", "model": "batch"})["streaming"])
            worker.handle({"type": "feed", "audio": "AAAAAA=="})
            self.assertEqual(worker.handle({"type": "finish"})["text"], "batch 1")
            self.assertEqual(engine.live, 1)
            worker.close()
            self.assertEqual(engine.live, 0)

    def test_input_limit_fails_instead_of_silently_truncating(self):
        registry = ModelRegistry(data=catalog())
        with patch.object(registry, "ensure_file", return_value=Path("batch")):
            worker = ModelWorker(FakeEngine(), registry)
            worker.handle({"type": "start", "model": "batch"})
            import base64
            pcm = array.array("f", [0] * 1601)
            with self.assertRaisesRegex(ValueError, "limit"):
                worker.handle({"type": "feed", "audio": base64.b64encode(pcm.tobytes()).decode()})
            worker.close()
