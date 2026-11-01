import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { commands, type RemoteModelInfo } from "@/bindings";
import { useSettings } from "@/hooks/useSettings";
import { Select } from "../ui/Select";
import { SettingsGroup } from "../ui/SettingsGroup";
import { SettingContainer } from "../ui/SettingContainer";
import { Button } from "../ui/Button";
import { LanguageSelector } from "./LanguageSelector";

export const RemoteModelSettings = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const [models, setModels] = useState<RemoteModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  const url = getSetting("remote_transcription_url");
  const current =
    getSetting("remote_transcription_model") ?? "parakeet-unified-en-0.6b";
  const model = models.find((entry) => entry.id === current);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);
    commands
      .getRemoteModels()
      .then((result) => {
        if (!active) return;
        if (result.status === "error") setError(result.error);
        else setModels(result.data);
      })
      .catch((error: unknown) => {
        if (active) setError(String(error));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [url, attempt]);

  return (
    <SettingsGroup title={t("settings.remote.title")}>
      <SettingContainer
        title={t("settings.remote.model")}
        description={t("settings.remote.description")}
        descriptionMode="inline"
        grouped
      >
        <Select
          value={loading ? null : current}
          options={models.map((entry) => ({
            value: entry.id,
            label: `${entry.name} · ${(entry.size_bytes / 1_000_000).toFixed(0)} MB`,
          }))}
          isLoading={loading}
          isClearable={false}
          disabled={
            loading ||
            Boolean(error) ||
            isUpdating("remote_transcription_model")
          }
          placeholder={
            loading ? t("settings.remote.loading") : t("settings.remote.choose")
          }
          onChange={(value) => {
            if (value) void updateSetting("remote_transcription_model", value);
          }}
        />
      </SettingContainer>
      {error && (
        <div className="px-4 py-3 flex items-center gap-3">
          <p role="alert" className="text-sm">
            {t("settings.remote.loadError")}
          </p>
          <Button
            variant="secondary"
            onClick={() => setAttempt((value) => value + 1)}
          >
            {t("settings.remote.retry")}
          </Button>
        </div>
      )}
      {model && !model.supports_streaming && (
        <p className="px-4 py-2 text-sm text-text/60">
          {t("settings.remote.batchOnly")}
        </p>
      )}
      {model && model.languages.length > 1 && (
        <LanguageSelector
          grouped
          supportedLanguages={model.languages}
          supportsLanguageDetection={model.supports_language_detection}
        />
      )}
    </SettingsGroup>
  );
};
