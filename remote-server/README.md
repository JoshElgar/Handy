# Handy remote Parakeet server

This single-process CPU server uses Handy's transcribe-cpp 0.2.4 binding and the Q8_0 model handy-computer/parakeet-unified-en-0.6b-gguf. The image builds the native library from pinned commit 4807edaf210d0d7e8a6f7fb2a44b65966a2797f0 and downloads the 731 MB model. It loads one model lazily and permits one transcription at a time.

## Build and run

```sh
docker build -t handy-parakeet-server .
docker run --rm -p 10000:10000 -e API_TOKEN='use-a-long-random-secret' handy-parakeet-server
```

Set API_TOKEN or API_TOKEN_FILE; if both are set they must match. The service listens on PORT (default 10000). /health is unauthenticated for Render health checks. Expose it only behind HTTPS.

## Streaming

Connect to wss://HOST/stream with Authorization: Bearer TOKEN.

1. Send text JSON {"type":"start","sample_rate":16000,"format":"f32le"}.
2. Wait for {"type":"ready"}. The server sends {"type":"progress"} every 10 seconds while it waits for audio and during native work.
3. Send binary little-endian float32 mono audio frames. Each frame must be 4-byte aligned and at most 64 KiB. The server processes frames in order; clients may send ahead while draining progress and partial messages concurrently.
4. After sending every frame, send {"type":"finish"}. The server drains all accepted audio, finalizes the model stream, emits {"type":"final","text":"..."}, then closes normally. Final text is the native stream's full hypothesis, including any tentative tail.

Each audio chunk gets a partial message with committed and tentative text. A malformed frame or protocol message produces an error and closes the stream. An occupied server emits {"type":"error","message":"busy"} then closes with code 1013. Disconnecting abandons the stream; the model is reusable after any native call already in progress has returned. Slow audio backlogs do not hit a WebSocket ping deadline; the client inactivity deadline still applies.

## History retry

For a complete recording, send POST /transcribe with Authorization: Bearer TOKEN, Content-Type: audio/wav, and a WAV body containing 16 kHz mono PCM16. The endpoint accepts files up to 128 MiB (about 70 minutes of raw PCM16; Handy history retries are expected to be shorter) and uses the same streaming decoder, feeding one-second chunks before finalizing. Success returns {"text":"..."}. Busy responses use HTTP 503 and Retry-After: 2.

## Configure the Handy fork

Quit Handy, then edit `~/Library/Application Support/com.pais.handy/settings_store.json`. Keep the other settings and add these keys inside its existing `settings` object:

```json
"remote_transcription_url": "https://YOUR_SERVICE.onrender.com",
"remote_transcription_api_keys": {
  "default": "YOUR_API_TOKEN"
}
```

Use the HTTPS service origin without a path, query, or fragment. The client uses the `default` token for both streaming and history retry. Store the file with user-only permissions and never share the token. Set `remote_transcription_url` to `null` or remove it to return to local transcription.

When a remote URL is configured, Handy skips loading its local speech model and reuses the existing live transcription overlay. The service uses the English Parakeet Unified model. Failed live requests keep the recorded audio in History so it can be retried. Audio waiting in the in-process remote queue uses local memory that grows with backlog; short streams are small, but avoid leaving a long queue accumulating.

After changing settings, reopen Handy. A fresh build from the fork is required; stock Handy does not use these settings.

## Checks

The image build runs the fake-native protocol tests before it downloads as a running service. To run them locally in a Python 3.12 environment with FastAPI, uvicorn, and httpx installed:

```sh
API_TOKEN=test-secret python -m unittest -v test_server
```

The tests cover auth, WebSocket protocol and audio validation, ordered complete frame delivery, final drain, periodic progress during native finalize, busy handling, failure/disconnect cleanup, and retry with audio longer than one minute. They use synthetic audio and a fake inference session.
