//! Remote Parakeet transport used by streaming dictation and history retry.
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::{mpsc, watch};
use tokio_tungstenite::{
    tungstenite::{client::IntoClientRequest, Message},
    WebSocketStream,
};

const MAX_AUDIO_FRAME_BYTES: usize = 64 * 1024;
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone)]
pub(crate) struct RemoteTranscriptionConfig {
    stream_url: String,
    transcribe_url: String,
    token: String,
}

impl RemoteTranscriptionConfig {
    pub(crate) fn from_settings(settings: &crate::settings::AppSettings) -> Result<Option<Self>> {
        let Some(base_url) = settings.remote_transcription_url.as_deref() else {
            return Ok(None);
        };
        if base_url.trim().is_empty() {
            return Ok(None);
        }

        let token = settings
            .remote_transcription_api_keys
            .get("default")
            .map(String::as_str)
            .unwrap_or_default();
        Self::from_url(base_url, token).map(Some)
    }

    fn from_url(base_url: &str, token: &str) -> Result<Self> {
        let mut url = reqwest::Url::parse(base_url.trim()).context("Invalid remote ASR URL")?;
        if url.scheme() != "https"
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.query().is_some()
            || url.fragment().is_some()
        {
            bail!(
                "Remote ASR URL must be an HTTPS base URL without credentials, query, or fragment"
            );
        }
        if token.trim().is_empty() {
            bail!("Remote ASR API token is missing");
        }

        let path = url.path().trim_end_matches('/').to_owned();
        url.set_path(&format!("{path}/stream"));
        let stream_url = url.as_str().replacen("https://", "wss://", 1);
        url.set_path(&format!("{path}/transcribe"));
        Ok(Self {
            stream_url,
            transcribe_url: url.to_string(),
            token: token.to_string(),
        })
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.token)
    }
}

#[derive(Debug)]
pub(crate) enum RemoteCommand {
    Audio(Vec<f32>),
    Finish,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ServerMessage {
    Ready,
    Progress,
    Partial {
        committed: String,
        tentative: String,
    },
    Final {
        text: String,
    },
    Error {
        message: String,
    },
}

pub(crate) async fn stream(
    config: &RemoteTranscriptionConfig,
    commands: mpsc::UnboundedReceiver<RemoteCommand>,
    mut cancel: watch::Receiver<bool>,
    mut emit: impl FnMut(String, String),
) -> Result<Option<String>> {
    if *cancel.borrow() {
        return Ok(None);
    }
    let cancelled = async {
        loop {
            if *cancel.borrow() || cancel.changed().await.is_err() {
                return;
            }
        }
    };
    tokio::select! {
        biased;
        result = async {
            let mut request = config.stream_url.as_str().into_client_request()
                .context("Failed to build remote stream request")?;
            request.headers_mut().insert(
                tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
                tokio_tungstenite::tungstenite::http::HeaderValue::from_str(&config.bearer())?,
            );
            let (socket, _) = tokio::time::timeout(
                Duration::from_secs(20), tokio_tungstenite::connect_async(request),
            ).await.context("Remote stream connection timed out")?
                .context("Failed to connect to remote transcription stream")?;
            run_stream(socket, commands, &mut emit, STREAM_IDLE_TIMEOUT).await
        } => result,
        _ = cancelled => Ok(None),
    }
}

async fn run_stream<S>(
    socket: WebSocketStream<S>,
    mut commands: mpsc::UnboundedReceiver<RemoteCommand>,
    emit: &mut impl FnMut(String, String),
    idle_timeout: Duration,
) -> Result<Option<String>>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut socket = socket;
    let start = Message::Text(
        json!({"type":"start","sample_rate":16000,"format":"f32le"})
            .to_string()
            .into(),
    );
    tokio::time::timeout(Duration::from_secs(20), socket.send(start))
        .await
        .context("Timed out starting remote transcription stream")?
        .context("Failed to start remote transcription stream")?;

    let mut last_inbound = tokio::time::Instant::now();
    loop {
        let inbound = tokio::time::timeout_at(last_inbound + idle_timeout, socket.next())
            .await
            .context(
                "Remote transcription stream made no progress before the inactivity deadline",
            )?;
        let Some(inbound) = inbound else {
            bail!("Remote transcription stream closed before ready");
        };
        let inbound = inbound.context("Failed to read remote transcription stream")?;
        last_inbound = tokio::time::Instant::now();
        match inbound {
            Message::Text(text) => match serde_json::from_str::<ServerMessage>(&text)
                .context("Invalid remote transcription response")?
            {
                ServerMessage::Ready => break,
                ServerMessage::Progress => {}
                ServerMessage::Partial {
                    committed,
                    tentative,
                } => emit(committed, tentative),
                ServerMessage::Final { .. } => bail!("Remote transcription finished before ready"),
                ServerMessage::Error { message } => bail!("Remote transcription failed: {message}"),
            },
            Message::Close(frame) => {
                bail!("Remote transcription stream closed before ready: {frame:?}")
            }
            Message::Ping(payload) => socket
                .send(Message::Pong(payload))
                .await
                .context("Failed to answer remote ping")?,
            Message::Binary(_) | Message::Pong(_) | Message::Frame(_) => {}
        }
    }

    let (mut writer, mut reader) = socket.split();
    let writer = async {
        while let Some(command) = commands.recv().await {
            match command {
                RemoteCommand::Audio(samples) => {
                    for chunk in samples.chunks(MAX_AUDIO_FRAME_BYTES / 4) {
                        let mut bytes = Vec::with_capacity(chunk.len() * 4);
                        for sample in chunk {
                            bytes.extend_from_slice(&sample.to_le_bytes());
                        }
                        writer
                            .send(Message::Binary(bytes.into()))
                            .await
                            .context("Failed to send audio to remote transcription stream")?;
                    }
                }
                RemoteCommand::Finish => {
                    writer
                        .send(Message::Text(json!({"type":"finish"}).to_string().into()))
                        .await
                        .context("Failed to finish remote transcription stream")?;
                    return Ok::<(), anyhow::Error>(());
                }
            }
        }
        bail!("Remote audio queue closed before finish")
    };
    tokio::pin!(writer);
    let mut writer_finished = false;
    loop {
        tokio::select! {
            result = &mut writer, if !writer_finished => {
                result?;
                writer_finished = true;
            }
            inbound = tokio::time::timeout_at(last_inbound + idle_timeout, reader.next()) => {
                let message = inbound
                    .context("Remote transcription stream made no progress before the inactivity deadline")?
                    .ok_or_else(|| anyhow!("Remote transcription stream closed before final result"))?
                    .context("Failed to read remote transcription stream")?;
                last_inbound = tokio::time::Instant::now();
                match message {
                    Message::Text(text) => {
                        match serde_json::from_str::<ServerMessage>(&text).context("Invalid remote transcription response")? {
                            ServerMessage::Ready => {},
                            ServerMessage::Progress => {},
                            ServerMessage::Partial { committed, tentative } => emit(committed, tentative),
                            ServerMessage::Final { text } => return Ok(Some(text)),
                            ServerMessage::Error { message } => bail!("Remote transcription failed: {message}"),
                        }
                    }
                    Message::Close(frame) => bail!("Remote transcription stream closed: {frame:?}"),
                    Message::Ping(_) => {},
                    Message::Binary(_) | Message::Pong(_) | Message::Frame(_) => {},
                }
            }
        }
    }
}

pub(crate) async fn transcribe(
    config: &RemoteTranscriptionConfig,
    audio: &[f32],
) -> Result<String> {
    let mut cursor = std::io::Cursor::new(Vec::with_capacity(44 + audio.len() * 2));
    let mut writer = hound::WavWriter::new(
        &mut cursor,
        hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in audio {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)?;
    }
    writer.finalize()?;

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(20))
        .timeout(Duration::from_secs(6_000))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("Failed to create remote transcription HTTP client")?;
    let response = client
        .post(&config.transcribe_url)
        .header(reqwest::header::AUTHORIZATION, config.bearer())
        .header(reqwest::header::CONTENT_TYPE, "audio/wav")
        .body(cursor.into_inner())
        .send()
        .await
        .context("Remote history retry request failed")?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        bail!("Remote history retry failed ({status}): {body}");
    }
    #[derive(Deserialize)]
    struct Response {
        text: String,
    }
    Ok(response
        .json::<Response>()
        .await
        .context("Invalid remote history retry response")?
        .text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::SinkExt;
    use serde_json::Value;
    use std::{net::SocketAddr, time::Duration};
    use tokio::net::{TcpListener, TcpStream};
    use tokio_tungstenite::{accept_async, tungstenite::Message};

    fn config_for_test(stream_url: String) -> RemoteTranscriptionConfig {
        RemoteTranscriptionConfig {
            stream_url,
            transcribe_url: String::new(),
            token: "test-secret".into(),
        }
    }
    async fn listener() -> (SocketAddr, TcpListener) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        (listener.local_addr().unwrap(), listener)
    }

    #[test]
    fn remote_config_derives_secure_endpoints_and_rejects_bad_settings() {
        let config =
            RemoteTranscriptionConfig::from_url("https://asr.example/api/", "secret").unwrap();
        assert_eq!(config.stream_url, "wss://asr.example/api/stream");
        assert_eq!(config.transcribe_url, "https://asr.example/api/transcribe");
        assert!(RemoteTranscriptionConfig::from_url("http://asr.example", "secret").is_err());
        assert!(RemoteTranscriptionConfig::from_url("https://u:p@asr.example", "secret").is_err());
        assert!(RemoteTranscriptionConfig::from_url("https://asr.example?x=1", "secret").is_err());
        assert!(RemoteTranscriptionConfig::from_url("https://asr.example", "  ").is_err());
    }

    #[tokio::test]
    async fn remote_fifo_sends_audio_then_finish_and_returns_final() {
        let (address, listener) = listener().await;
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(tcp).await.unwrap();
            let start = ws.next().await.unwrap().unwrap().into_text().unwrap();
            assert_eq!(
                serde_json::from_str::<Value>(&start).unwrap(),
                json!({"type":"start","sample_rate":16000,"format":"f32le"})
            );
            ws.send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            let mut samples = Vec::new();
            loop {
                match ws.next().await.unwrap().unwrap() {
                    Message::Binary(bytes) => {
                        assert!(bytes.len() <= MAX_AUDIO_FRAME_BYTES);
                        samples.extend(
                            bytes
                                .chunks_exact(4)
                                .map(|b| f32::from_le_bytes(b.try_into().unwrap())),
                        );
                    }
                    Message::Text(text) if text.contains("finish") => {
                        assert_eq!(samples, (0..20_000).map(|n| n as f32).collect::<Vec<_>>());
                        ws.send(Message::Text(
                            json!({"type":"final","text":"complete"}).to_string().into(),
                        ))
                        .await
                        .unwrap();
                        return;
                    }
                    message => panic!("unexpected remote message: {message:?}"),
                }
            }
        });
        let (tx, rx) = mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        tx.send(RemoteCommand::Audio(
            (0..20_000).map(|n| n as f32).collect(),
        ))
        .unwrap();
        tx.send(RemoteCommand::Finish).unwrap();
        let result = stream(
            &config_for_test(format!("ws://{address}/stream")),
            rx,
            cancel_rx,
            |_, _| {},
        )
        .await
        .unwrap();
        assert_eq!(result.as_deref(), Some("complete"));
        drop(cancel_tx);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn remote_final_wait_uses_progress_as_idle_heartbeat() {
        let (address, listener) = listener().await;
        let (ping_received_tx, ping_received_rx) = tokio::sync::oneshot::channel::<()>();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(tcp).await.unwrap();
            let _ = ws.next().await.unwrap().unwrap();
            ws.send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            assert!(matches!(
                ws.next().await.unwrap().unwrap(),
                Message::Text(_)
            ));
            ws.send(Message::Ping(b"keepalive".to_vec().into()))
                .await
                .unwrap();
            assert!(matches!(
                ws.next().await.unwrap().unwrap(),
                Message::Pong(_)
            ));
            let _ = ping_received_tx.send(());
            for _ in 0..3 {
                tokio::time::sleep(Duration::from_millis(40)).await;
                ws.send(Message::Text(json!({"type":"progress"}).to_string().into()))
                    .await
                    .unwrap();
            }
            ws.send(Message::Text(
                json!({"type":"final","text":"slow final"})
                    .to_string()
                    .into(),
            ))
            .await
            .unwrap();
        });
        let (tx, rx) = mpsc::unbounded_channel();
        let (_cancel_tx, _cancel_rx) = watch::channel(false);
        tx.send(RemoteCommand::Finish).unwrap();
        let mut noop = |_, _| {};
        let tcp = TcpStream::connect(address).await.unwrap();
        let socket = tokio_tungstenite::client_async(format!("ws://{address}/stream"), tcp)
            .await
            .unwrap()
            .0;
        let result = tokio::time::timeout(
            Duration::from_millis(250),
            run_stream(socket, rx, &mut noop, Duration::from_millis(75)),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result.as_deref(), Some("slow final"));
        tokio::time::timeout(Duration::from_secs(1), ping_received_rx)
            .await
            .unwrap()
            .unwrap();
        server.await.unwrap();
    }

    #[tokio::test]
    async fn remote_disconnect_fails_and_out_of_band_cancel_interrupts_backlog() {
        let (address, listener) = listener().await;
        let (audio_tx, audio_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(tcp).await.unwrap();
            let _ = ws.next().await.unwrap().unwrap();
            ws.send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            assert!(matches!(
                ws.next().await.unwrap().unwrap(),
                Message::Binary(_)
            ));
            let _ = audio_tx.send(());
            tokio::time::sleep(Duration::from_millis(300)).await;
        });
        let (tx, rx) = mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        tx.send(RemoteCommand::Audio(vec![0.25; 1_000_000]))
            .unwrap();
        tx.send(RemoteCommand::Finish).unwrap();
        let config = config_for_test(format!("ws://{address}/stream"));
        let task = tokio::spawn(async move { stream(&config, rx, cancel_rx, |_, _| {}).await });
        tokio::time::timeout(Duration::from_secs(1), audio_rx)
            .await
            .unwrap()
            .unwrap();
        cancel_tx.send(true).unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .is_none());
        server.abort();

        let (address2, listener2) = TcpListener::bind("127.0.0.1:0")
            .await
            .map(|listener| (listener.local_addr().unwrap(), listener))
            .unwrap();
        let server = tokio::spawn(async move {
            let (tcp, _) = listener2.accept().await.unwrap();
            let mut ws = accept_async(tcp).await.unwrap();
            let _ = ws.next().await.unwrap().unwrap();
            ws.send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            drop(ws);
        });
        let (tx, rx) = mpsc::unbounded_channel();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        tx.send(RemoteCommand::Finish).unwrap();
        let result = stream(
            &config_for_test(format!("ws://{address2}/stream")),
            rx,
            cancel_rx,
            |_, _| {},
        )
        .await;
        assert!(result.is_err());
        server.await.unwrap();
    }
}
