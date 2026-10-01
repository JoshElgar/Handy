# Handy remote Parakeet server

This CPU server with one persistent model subprocess uses Handy's transcribe-cpp 0.2.4 binding and the Q8_0 model handy-computer/parakeet-unified-en-0.6b-gguf. The image builds the native library from pinned commit 4807edaf210d0d7e8a6f7fb2a44b65966a2797f0 and downloads the 731 MB model. It loads one model lazily and processes the newest transcription, using two inference threads on the two-CPU Render instance.

The CPU backend is built as portable ISA-specific modules and selects the best supported variant at runtime. The x64 baseline remains available for older hosts.

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

Each audio chunk gets a partial message with committed and tentative text. A malformed frame or protocol message produces an error and closes the stream. A valid new start replaces unfinished work globally (this deployment is for one user). The server kills and reaps the old model process, discards its audio and returns {"type":"error","message":"replaced by a newer recording"} to the old stream. The new stream starts a fresh model. Successful recordings reuse the loaded process. For Cancel, send {"type":"cancel"} and close the socket. The server acknowledges {"type":"cancelled"}, discards queued audio, and notices disconnects even during inference. Cancel or disconnect kills unfinished model work, even if inference is hung. Cancellation is acknowledged after process termination.

The server checks WebSocket ping/pong every 20 seconds with a 20-second response deadline. The client continues answering while waiting for the final transcript. These checks detect a dead connection, not slow transcription. Received audio is limited to a 128 MiB backlog or 8192 queued messages; exceeding either closes the stream with an explicit error. Ordinary Finish still drains every accepted frame in order.

Logs identify model process starts/stops and command start/return times. The subprocess uses a private stdin/stdout protocol; native diagnostics go to stderr. No additional service or dependency is needed.

## History retry

For a complete recording, send POST /transcribe with Authorization: Bearer TOKEN, Content-Type: audio/wav, and a WAV body containing 16 kHz mono PCM16. The endpoint accepts files up to 128 MiB (about 70 minutes of raw PCM16; Handy history retries are expected to be shorter) and uses the same streaming decoder, feeding one-second chunks before finalizing. Success returns {"text":"..."}. History requests share the newest-request policy: a newer request replaces unfinished work, and a replaced history request receives HTTP 409.

## Configure the Handy fork

Quit Handy Remote, then edit `~/Library/Application Support/com.joshelgar.handy.remote/settings_store.json`. Keep the other settings and add these keys inside its existing `settings` object:

```json
"remote_transcription_url": "https://YOUR_SERVICE.onrender.com",
"remote_transcription_api_keys": {
  "default": "YOUR_API_TOKEN"
}
```

Use the HTTPS service origin without a path, query, or fragment. The client uses the `default` token for both streaming and history retry. Store the file with user-only permissions and never share the token. Set `remote_transcription_url` to `null` or remove it to return to local transcription.

When a remote URL is configured, Handy skips loading its local speech model and reuses the existing live transcription overlay. The service uses the English Parakeet Unified model. Failed live requests keep the recorded audio in History so it can be retried. Audio waiting in the in-process remote queue uses local memory that grows with backlog; short streams are small, but avoid leaving a long queue accumulating.

After changing settings, reopen Handy Remote. A fresh build from the fork is required; stock Handy does not use these settings. The fork stores settings and History separately; remote transcription does not load the local speech model. Since macOS treats Handy Remote as a separate app, grant it fresh Accessibility and microphone permissions on first use. The original Handy app may remain installed, but avoid enabling autostart in both apps because they share global keyboard shortcuts.

## Checks

The image build runs the fake-native protocol tests before it downloads as a running service. To run them locally in a Python 3.12 environment with FastAPI, uvicorn, and httpx installed:

```sh
API_TOKEN=test-secret python -m unittest -v test_server
```

The tests cover auth, WebSocket protocol and audio validation, ordered complete frame delivery, final drain, periodic progress during native finalize, replacement of a hung process, successful process reuse, cancellation/disconnect, crash recovery, history retry and validation. They use synthetic audio and a real subprocess with a fake model protocol.
