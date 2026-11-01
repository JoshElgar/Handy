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
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(35);
const STREAM_FINAL_TIMEOUT: Duration = Duration::from_secs(90);
pub(crate) const MODEL_OPERATION_TIMEOUT: Duration = Duration::from_secs(600);

#[derive(Clone)]
pub(crate) struct RemoteTranscriptionConfig {
    stream_url: String,
    transcribe_url: String,
    token: String,
    model: Option<String>,
    language: String,
    supported_languages: Vec<String>,
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
        let mut config = Self::from_url(base_url, token)?;
        config.model = settings.remote_transcription_model.clone();
        let slug = config
            .model
            .as_deref()
            .unwrap_or("parakeet-unified-en-0.6b");
        let repo = format!("handy-computer/{slug}-gguf");
        let descriptor = crate::catalog::CATALOG.iter().find(|model| {
            matches!(&model.source, crate::managers::model::ModelSource::HuggingFace { repo_id, .. } if repo_id == &repo)
        }).ok_or_else(|| anyhow!("Unknown remote transcription model: {slug}"))?;
        config.supported_languages = descriptor.caps.languages.clone().unwrap_or_default();
        config.language = crate::managers::model::effective_language(
            &settings.selected_language,
            &config.supported_languages,
            descriptor.caps.supports_language_detect.unwrap_or(false),
        );
        Ok(Some(config))
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
            model: None,
            language: "auto".into(),
            supported_languages: vec!["en".into()],
        })
    }

    fn bearer(&self) -> String {
        format!("Bearer {}", self.token)
    }

    fn start_message(&self) -> serde_json::Value {
        let mut message = json!({"type":"start","sample_rate":16000,"format":"f32le"});
        if let Some(model) = &self.model {
            message["model"] = json!(model);
            message["language"] = json!(self.language);
        }
        message
    }

    fn transcribe_request_url(&self) -> reqwest::Url {
        let mut url = reqwest::Url::parse(&self.transcribe_url).expect("validated URL");
        if let Some(model) = &self.model {
            url.query_pairs_mut()
                .append_pair("model", model)
                .append_pair("language", &self.language);
        }
        url
    }

    pub(crate) fn supported_languages(&self) -> Vec<String> {
        self.supported_languages.clone()
    }

    pub(crate) fn output_language(&self) -> crate::audio_toolkit::OutputLanguageEvidence {
        use crate::audio_toolkit::OutputLanguageEvidence;
        // Some multilingual models ignore language hints. Let the existing
        // text-language detection resolve their output instead of assuming it.
        if self.supported_languages.len() == 1 {
            OutputLanguageEvidence::ModelConstrained(self.supported_languages[0].clone())
        } else {
            OutputLanguageEvidence::Unknown
        }
    }
}

#[derive(Clone, serde::Serialize, Deserialize, specta::Type)]
pub struct RemoteModelInfo {
    pub id: String,
    pub name: String,
    pub languages: Vec<String>,
    pub supports_streaming: bool,
    pub supports_language_detection: bool,
    pub size_bytes: u64,
}

pub(crate) async fn list_models(
    config: &RemoteTranscriptionConfig,
) -> Result<Vec<RemoteModelInfo>> {
    let mut url = reqwest::Url::parse(&config.transcribe_url)?;
    let base = url.path().trim_end_matches("/transcribe").to_owned();
    url.set_path(&format!("{base}/models"));
    reqwest::Client::builder()
        .timeout(Duration::from_secs(20))
        .redirect(reqwest::redirect::Policy::none())
        .build()?
        .get(url)
        .header(reqwest::header::AUTHORIZATION, config.bearer())
        .send()
        .await?
        .error_for_status()?
        .json()
        .await
        .context("Invalid remote model catalogue")
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
    Cancelled,
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
    let mut request = config
        .stream_url
        .as_str()
        .into_client_request()
        .context("Failed to build remote stream request")?;
    request.headers_mut().insert(
        tokio_tungstenite::tungstenite::http::header::AUTHORIZATION,
        tokio_tungstenite::tungstenite::http::HeaderValue::from_str(&config.bearer())?,
    );
    let deadline = tokio::time::Instant::now() + Duration::from_secs(20);
    let mut attempt = 0;
    let socket = loop {
        attempt += 1;
        let result = tokio::select! {
            biased;
            _ = wait_for_cancel(&mut cancel) => return Ok(None),
            result = tokio::time::timeout_at(deadline, tokio_tungstenite::connect_async(request.clone())) => {
                result.context("Remote stream connection timed out")?
            }
        };
        match result {
            Ok((socket, _)) => break socket,
            Err(error) => {
                let retryable = match &error {
                    tokio_tungstenite::tungstenite::Error::Io(_) => true,
                    tokio_tungstenite::tungstenite::Error::Http(response) => {
                        response.status().is_server_error() || response.status().as_u16() == 429
                    }
                    _ => false,
                };
                if !retryable || attempt >= 3 {
                    return Err(error).context("Failed to connect to remote transcription stream");
                }
                log::warn!("Remote connection attempt {attempt} failed; retrying before sending audio: {error}");
                tokio::select! {
                    biased;
                    _ = wait_for_cancel(&mut cancel) => return Ok(None),
                    _ = tokio::time::sleep_until(deadline) => bail!("Remote stream connection timed out: {error}"),
                    _ = tokio::time::sleep(Duration::from_millis(500)) => {}
                }
            }
        }
    };
    run_stream(
        socket,
        commands,
        &mut emit,
        STREAM_IDLE_TIMEOUT,
        if config.model.is_some() {
            MODEL_OPERATION_TIMEOUT
        } else {
            STREAM_FINAL_TIMEOUT
        },
        cancel,
        config.start_message(),
    )
    .await
}

async fn wait_for_cancel(cancel: &mut watch::Receiver<bool>) {
    while !*cancel.borrow() {
        if cancel.changed().await.is_err() {
            break;
        }
    }
}

async fn cancel_socket<S>(socket: &mut S)
where
    S: futures_util::Sink<Message> + Unpin,
{
    // Cancellation abandons queued audio; a bounded send cannot keep the app busy.
    let _ = tokio::time::timeout(Duration::from_secs(2), async {
        socket
            .send(Message::Text(json!({"type":"cancel"}).to_string().into()))
            .await?;
        socket.send(Message::Close(None)).await
    })
    .await;
}

async fn run_stream<S>(
    socket: WebSocketStream<S>,
    mut commands: mpsc::UnboundedReceiver<RemoteCommand>,
    emit: &mut impl FnMut(String, String),
    idle_timeout: Duration,
    final_timeout: Duration,
    mut cancel: watch::Receiver<bool>,
    start_message: serde_json::Value,
) -> Result<Option<String>>
where
    S: tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
{
    let mut socket = socket;
    let ready_timeout = if start_message.get("model").is_some() {
        MODEL_OPERATION_TIMEOUT
    } else {
        idle_timeout
    };
    let start = Message::Text(start_message.to_string().into());
    tokio::select! {
        biased;
        _ = wait_for_cancel(&mut cancel) => return Ok(None),
        result = tokio::time::timeout(idle_timeout, socket.send(start)) => {
            result.context("Timed out starting remote transcription stream")?
                .context("Failed to start remote transcription stream")?;
        }
    }

    let mut last_inbound = tokio::time::Instant::now();
    let ready_deadline = last_inbound + ready_timeout;
    loop {
        let inbound = tokio::select! {
            biased;
            _ = wait_for_cancel(&mut cancel) => {
                cancel_socket(&mut socket).await;
                return Ok(None);
            }
            result = tokio::time::timeout_at((last_inbound + idle_timeout).min(ready_deadline), socket.next()) => {
                result.context("Remote transcription server did not become ready; check your connection and try again")?
            }
        };
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
                ServerMessage::Cancelled => return Ok(None),
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
            Message::Ping(payload) => {
                tokio::time::timeout_at(ready_deadline, socket.send(Message::Pong(payload)))
                    .await
                    .context("Remote connection stalled before the server became ready")?
                    .context("Failed to answer remote ping")?
            }
            Message::Binary(_) | Message::Pong(_) | Message::Frame(_) => {}
        }
    }

    let (mut writer, mut reader) = socket.split();
    let (pong_tx, mut pong_rx) = mpsc::unbounded_channel();
    let (finished_tx, mut finished_rx) = watch::channel(None);
    let writer = async {
        let mut finished = false;
        loop {
            tokio::select! {
                biased;
                _ = wait_for_cancel(&mut cancel) => {
                    cancel_socket(&mut writer).await;
                    return Ok::<(), anyhow::Error>(());
                }
                Some(payload) = pong_rx.recv() => {
                    tokio::time::timeout(idle_timeout, writer.send(Message::Pong(payload))).await
                        .context("Remote connection stalled while answering ping")?
                        .context("Failed to answer remote ping")?;
                }
                command = commands.recv(), if !finished => {
                    match command {
                        Some(RemoteCommand::Audio(samples)) => {
                            for chunk in samples.chunks(MAX_AUDIO_FRAME_BYTES / 4) {
                                let bytes: Vec<u8> = chunk.iter().flat_map(|sample| sample.to_le_bytes()).collect();
                                tokio::select! {
                                    biased;
                                    _ = wait_for_cancel(&mut cancel) => {
                                        cancel_socket(&mut writer).await;
                                        return Ok(());
                                    }
                                    result = tokio::time::timeout(idle_timeout, writer.send(Message::Binary(bytes.into()))) => {
                                        result.context("Remote connection stalled while sending audio")?
                                            .context("Failed to send audio to remote transcription stream")?;
                                    }
                                }
                            }
                        }
                        Some(RemoteCommand::Finish) => {
                            let _ = finished_tx.send(Some(tokio::time::Instant::now() + final_timeout));
                            tokio::time::timeout(idle_timeout, writer.send(Message::Text(json!({"type":"finish"}).to_string().into()))).await
                                .context("Remote connection stalled while finishing")?
                                .context("Failed to finish remote transcription stream")?;
                            finished = true;
                        }
                        None => bail!("Remote audio queue closed before finish"),
                    }
                }
            }
        }
    };
    tokio::pin!(writer);
    loop {
        let final_deadline = *finished_rx.borrow();
        tokio::select! {
            _ = finished_rx.changed() => {},
            _ = async {
                match final_deadline {
                    Some(deadline) => tokio::time::sleep_until(deadline).await,
                    None => std::future::pending().await,
                }
            } => bail!("Remote transcription did not finish within the deadline; your audio is saved in History for retry"),
            result = &mut writer => {
                result?;
                return Ok(None);
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
                            ServerMessage::Cancelled => return Ok(None),
                            ServerMessage::Partial { committed, tentative } => emit(committed, tentative),
                            ServerMessage::Final { text } => return Ok(Some(text)),
                            ServerMessage::Error { message } => bail!("Remote transcription failed: {message}"),
                        }
                    }
                    Message::Close(frame) => bail!("Remote transcription stream closed: {frame:?}"),
                    Message::Ping(payload) => { let _ = pong_tx.send(payload); },
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
        .timeout(MODEL_OPERATION_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("Failed to create remote transcription HTTP client")?;
    let response = client
        .post(config.transcribe_request_url())
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
            model: None,
            language: "auto".into(),
            supported_languages: vec!["en".into()],
        }
    }
    async fn listener() -> (SocketAddr, TcpListener) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        (listener.local_addr().unwrap(), listener)
    }

    #[test]
    fn model_selection_and_language_are_snapshotted_in_request_config() {
        let mut settings = crate::settings::get_default_settings();
        settings.remote_transcription_url = Some("https://asr.example".into());
        settings
            .remote_transcription_api_keys
            .insert("default".into(), "secret".into());
        settings.remote_transcription_model = Some("moonshine-streaming-small".into());
        settings.selected_language = "en".into();
        let config = RemoteTranscriptionConfig::from_settings(&settings)
            .unwrap()
            .unwrap();
        settings.remote_transcription_model = Some("whisper-medium".into());
        assert_eq!(config.start_message()["model"], "moonshine-streaming-small");
        assert_eq!(config.start_message()["language"], "en");
        assert_eq!(config.supported_languages(), vec!["en".to_string()]);
        assert_eq!(
            config
                .transcribe_request_url()
                .query_pairs()
                .find(|(k, _)| k == "model")
                .unwrap()
                .1,
            "moonshine-streaming-small"
        );
    }

    #[test]
    fn multilingual_selection_does_not_assume_english_output() {
        let mut settings = crate::settings::get_default_settings();
        settings.remote_transcription_url = Some("https://asr.example".into());
        settings
            .remote_transcription_api_keys
            .insert("default".into(), "secret".into());
        settings.remote_transcription_model = Some("whisper-medium".into());
        let config = RemoteTranscriptionConfig::from_settings(&settings)
            .unwrap()
            .unwrap();
        assert_eq!(
            config.output_language(),
            crate::audio_toolkit::text::OutputLanguageEvidence::Unknown
        );
    }

    #[tokio::test]
    async fn remote_connection_retries_gateway_failure_without_losing_audio() {
        let (address, listener) = listener().await;
        let server = tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut request = [0; 2048];
            tcp.read(&mut request).await.unwrap();
            tcp.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            drop(tcp);
            let (tcp, _) = listener.accept().await.unwrap();
            let mut socket = accept_async(tcp).await.unwrap();
            assert!(socket
                .next()
                .await
                .unwrap()
                .unwrap()
                .into_text()
                .unwrap()
                .contains("start"));
            socket
                .send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            assert_eq!(
                socket.next().await.unwrap().unwrap().into_data().as_ref(),
                0.25_f32.to_le_bytes()
            );
            assert!(socket
                .next()
                .await
                .unwrap()
                .unwrap()
                .into_text()
                .unwrap()
                .contains("finish"));
            socket
                .send(Message::Text(
                    json!({"type":"final","text":"recovered"})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
        });
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(RemoteCommand::Audio(vec![0.25])).unwrap();
        tx.send(RemoteCommand::Finish).unwrap();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let result = tokio::time::timeout(
            Duration::from_secs(3),
            stream(
                &config_for_test(format!("ws://{address}/stream")),
                rx,
                cancel_rx,
                |_, _| {},
            ),
        )
        .await
        .unwrap()
        .unwrap();
        assert_eq!(result.as_deref(), Some("recovered"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn remote_connection_auth_failure_is_not_retried() {
        let (address, listener) = listener().await;
        let server = tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut request = [0; 2048];
            tcp.read(&mut request).await.unwrap();
            tcp.write_all(b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            drop(tcp);
            assert!(
                tokio::time::timeout(Duration::from_millis(750), listener.accept())
                    .await
                    .is_err()
            );
        });
        let (_tx, rx) = mpsc::unbounded_channel();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let error = stream(
            &config_for_test(format!("ws://{address}/stream")),
            rx,
            cancel_rx,
            |_, _| {},
        )
        .await
        .unwrap_err();
        assert!(format!("{error:#}").contains("401"));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn remote_connection_cancel_interrupts_retry_delay() {
        let (address, listener) = listener().await;
        let (failed_tx, failed_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut tcp, _) = listener.accept().await.unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut request = [0; 2048];
            tcp.read(&mut request).await.unwrap();
            tcp.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                .await
                .unwrap();
            drop(tcp);
            failed_tx.send(()).unwrap();
            assert!(
                tokio::time::timeout(Duration::from_millis(750), listener.accept())
                    .await
                    .is_err()
            );
        });
        let (_tx, rx) = mpsc::unbounded_channel();
        let (cancel_tx, cancel_rx) = watch::channel(false);
        let config = config_for_test(format!("ws://{address}/stream"));
        let client = tokio::spawn(async move { stream(&config, rx, cancel_rx, |_, _| {}).await });
        failed_rx.await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel_tx.send(true).unwrap();
        assert!(tokio::time::timeout(Duration::from_millis(250), client)
            .await
            .unwrap()
            .unwrap()
            .unwrap()
            .is_none());
        server.await.unwrap();
    }

    #[tokio::test]
    async fn remote_connection_stops_after_three_gateway_failures() {
        let (address, listener) = listener().await;
        let server = tokio::spawn(async move {
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            for _attempt in 0..3 {
                let (mut tcp, _) = listener.accept().await.unwrap();
                let mut request = [0; 2048];
                tcp.read(&mut request).await.unwrap();
                tcp.write_all(b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\n\r\n")
                    .await
                    .unwrap();
            }
            assert!(
                tokio::time::timeout(Duration::from_millis(750), listener.accept())
                    .await
                    .is_err()
            );
        });
        let (_tx, rx) = mpsc::unbounded_channel();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let error = tokio::time::timeout(
            Duration::from_secs(3),
            stream(
                &config_for_test(format!("ws://{address}/stream")),
                rx,
                cancel_rx,
                |_, _| {},
            ),
        )
        .await
        .unwrap()
        .unwrap_err();
        assert!(format!("{error:#}").contains("503"));
        server.await.unwrap();
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
    async fn remote_recovery_blackholed_stream_times_out_and_next_session_succeeds() {
        let (address, listener) = listener().await;
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut stalled = accept_async(tcp).await.unwrap();
            stalled.next().await.unwrap().unwrap();
            stalled
                .send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            let (tcp, _) = listener.accept().await.unwrap();
            let mut recovered = accept_async(tcp).await.unwrap();
            recovered.next().await.unwrap().unwrap();
            recovered
                .send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            recovered.next().await.unwrap().unwrap();
            recovered
                .send(Message::Text(
                    json!({"type":"final","text":"recovered"})
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
        });
        for succeeds in [false, true] {
            let tcp = TcpStream::connect(address).await.unwrap();
            let socket = tokio_tungstenite::client_async(format!("ws://{address}/stream"), tcp)
                .await
                .unwrap()
                .0;
            let (tx, rx) = mpsc::unbounded_channel();
            tx.send(RemoteCommand::Finish).unwrap();
            let (_cancel_tx, cancel_rx) = watch::channel(false);
            let result = run_stream(
                socket,
                rx,
                &mut |_, _| {},
                Duration::from_millis(100),
                Duration::from_secs(1),
                cancel_rx,
                json!({"type":"start","sample_rate":16000,"format":"f32le"}),
            )
            .await;
            if succeeds {
                assert_eq!(result.unwrap().as_deref(), Some("recovered"));
            } else {
                assert!(format!("{:#}", result.unwrap_err()).contains("inactivity"));
            }
        }
        server.await.unwrap();
    }

    #[tokio::test]
    async fn remote_recovery_progress_cannot_keep_unready_session_alive_forever() {
        let (address, listener) = listener().await;
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(tcp).await.unwrap();
            ws.next().await.unwrap().unwrap();
            for _ in 0..20 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                if ws
                    .send(Message::Text(json!({"type":"progress"}).to_string().into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
        let tcp = TcpStream::connect(address).await.unwrap();
        let socket = tokio_tungstenite::client_async(format!("ws://{address}/stream"), tcp)
            .await
            .unwrap()
            .0;
        let (_tx, rx) = mpsc::unbounded_channel();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let result = tokio::time::timeout(
            Duration::from_millis(150),
            run_stream(
                socket,
                rx,
                &mut |_, _| {},
                Duration::from_millis(50),
                Duration::from_secs(1),
                cancel_rx,
                json!({"type":"start","sample_rate":16000,"format":"f32le"}),
            ),
        )
        .await;
        assert!(
            result.is_ok(),
            "progress must not extend the ready deadline"
        );
        assert!(result.unwrap().is_err());
        server.abort();
    }

    #[tokio::test]
    async fn remote_recovery_progress_cannot_keep_final_loading_forever() {
        let (address, listener) = listener().await;
        let server = tokio::spawn(async move {
            let (tcp, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(tcp).await.unwrap();
            ws.next().await.unwrap().unwrap();
            ws.send(Message::Text(json!({"type":"ready"}).to_string().into()))
                .await
                .unwrap();
            ws.next().await.unwrap().unwrap();
            for _ in 0..20 {
                tokio::time::sleep(Duration::from_millis(10)).await;
                if ws
                    .send(Message::Text(json!({"type":"progress"}).to_string().into()))
                    .await
                    .is_err()
                {
                    break;
                }
            }
        });
        let tcp = TcpStream::connect(address).await.unwrap();
        let socket = tokio_tungstenite::client_async(format!("ws://{address}/stream"), tcp)
            .await
            .unwrap()
            .0;
        let (tx, rx) = mpsc::unbounded_channel();
        tx.send(RemoteCommand::Finish).unwrap();
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        let result = tokio::time::timeout(
            Duration::from_millis(150),
            run_stream(
                socket,
                rx,
                &mut |_, _| {},
                Duration::from_millis(100),
                Duration::from_millis(50),
                cancel_rx,
                json!({"type":"start","sample_rate":16000,"format":"f32le"}),
            ),
        )
        .await;
        assert!(
            result.is_ok(),
            "progress must not extend the final deadline"
        );
        assert!(result.unwrap().is_err());
        server.abort();
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
                                .as_ref()
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|bytes| f32::from_le_bytes(*bytes)),
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
        let (_cancel_tx, cancel_rx) = watch::channel(false);
        tx.send(RemoteCommand::Finish).unwrap();
        let mut noop = |_, _| {};
        let tcp = TcpStream::connect(address).await.unwrap();
        let socket = tokio_tungstenite::client_async(format!("ws://{address}/stream"), tcp)
            .await
            .unwrap()
            .0;
        let result = tokio::time::timeout(
            Duration::from_millis(250),
            run_stream(
                socket,
                rx,
                &mut noop,
                Duration::from_millis(75),
                Duration::from_secs(1),
                cancel_rx,
                json!({"type":"start","sample_rate":16000,"format":"f32le"}),
            ),
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
            loop {
                match ws.next().await.unwrap().unwrap() {
                    Message::Text(text) if text.contains("cancel") => {
                        ws.send(Message::Text(
                            json!({"type":"cancelled"}).to_string().into(),
                        ))
                        .await
                        .unwrap();
                        assert!(matches!(
                            ws.next().await.unwrap().unwrap(),
                            Message::Close(_)
                        ));
                        return;
                    }
                    Message::Binary(_) => {}
                    message => panic!("unexpected cancellation message: {message:?}"),
                }
            }
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
        tokio::time::timeout(Duration::from_secs(2), server)
            .await
            .unwrap()
            .unwrap();

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
