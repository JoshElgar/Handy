import array
import base64
import json
import os
import sys


def run():
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    import transcribe_cpp

    model = None
    session = None
    decoder = None
    for line in sys.stdin:
        command = json.loads(line)
        if command["type"] == "start":
            if model is None:
                model = transcribe_cpp.Model(os.environ["MODEL_PATH"], backend="cpu")
            session = model.session(n_threads=2)
            decoder = session.stream()
            reply = {"type": "ready"}
        elif command["type"] == "feed":
            pcm = array.array("f")
            pcm.frombytes(base64.b64decode(command["audio"]))
            if sys.byteorder == "big":
                pcm.byteswap()
            decoder.feed(pcm)
            text = decoder.text()
            reply = {
                "type": "partial",
                "committed": text.committed,
                "tentative": text.tentative,
            }
        elif command["type"] == "finish":
            decoder.finalize()
            reply = {"type": "final", "text": decoder.text().full.strip()}
            decoder.reset()
            session.close()
            decoder = session = None
        else:
            raise ValueError("Invalid worker command")
        protocol.write(json.dumps(reply) + "\n")
    if model is not None:
        model.close()


if __name__ == "__main__":
    run()
