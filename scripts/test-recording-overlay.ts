import assert from "node:assert/strict";
import { mock } from "bun:test";

const listeners = new Map<string, (event: { payload: unknown }) => unknown>();
const state: unknown[] = [];
let nextState = 0;
let effect: (() => unknown) | undefined;
let resolveSettings!: (value: unknown) => void;
const pendingSettings = new Promise((resolve) => {
  resolveSettings = resolve;
});
let unlistened = 0;
const listen = async (
  name: string,
  callback: (event: { payload: unknown }) => unknown,
) => {
  listeners.set(name, callback);
  return () => {
    unlistened += 1;
  };
};

mock.module("react", () => ({
  default: {},
  useState: (initial: unknown) => {
    const index = nextState++;
    state[index] = initial;
    return [
      initial,
      (value: unknown) => {
        state[index] =
          typeof value === "function" ? value(state[index]) : value;
      },
    ];
  },
  useRef: (current: unknown) => ({ current }),
  useEffect: (callback: () => unknown) => {
    effect ??= callback;
  },
  useLayoutEffect: () => {},
}));
mock.module("react/jsx-dev-runtime", () => ({ jsxDEV: () => null }));
mock.module("react/jsx-runtime", () => ({ jsx: () => null, jsxs: () => null }));
mock.module("@tauri-apps/api/event", () => ({ listen }));
mock.module("react-i18next", () => ({
  useTranslation: () => ({ t: () => "" }),
}));
mock.module("@/bindings", () => ({
  commands: {
    getAppSettings: () => pendingSettings,
    cancelOperation: async () => {},
  },
  events: {
    streamTextEvent: {
      listen: (callback: Parameters<typeof listen>[1]) =>
        listen("text", callback),
    },
    streamPhaseEvent: {
      listen: (callback: Parameters<typeof listen>[1]) =>
        listen("phase", callback),
    },
  },
}));
mock.module("@/i18n", () => ({
  default: { language: "en" },
  syncLanguageFromSettings: async () => {},
}));
mock.module("@/lib/utils/rtl", () => ({ getLanguageDirection: () => "ltr" }));

const { default: RecordingOverlay } = await import(
  "../src/overlay/RecordingOverlay"
);
RecordingOverlay({});
const cleanup = effect?.();

assert.equal(
  listeners.size,
  6,
  "register every listener before awaiting any registration",
);
const showing = listeners.get("show-overlay")?.({ payload: "streaming" });
assert.equal(state[0], true, "show recording without waiting for settings I/O");
assert.equal(state[1], "streaming");
assert.equal(state[2], false);
listeners.get("recording-ready")?.({ payload: undefined });
assert.equal(state[2], true, "readiness must not be lost during settings I/O");
listeners.get("hide-overlay")?.({ payload: undefined });
assert.equal(state[0], false);
assert.equal(state[2], false);
resolveSettings({ status: "ok", data: { overlay_position: "top" } });
await showing;
assert.equal(
  state[0],
  false,
  "late settings must not resurrect a hidden overlay",
);

await listeners.get("show-overlay")?.({ payload: "streaming" });
listeners.get("mic-level")?.({ payload: Array(16).fill(0) });
assert.equal(
  state[2],
  true,
  "real sample events recover a missed one-shot readiness event, including silence",
);
assert.equal(
  typeof cleanup,
  "function",
  "unmount must dispose native listeners",
);
if (typeof cleanup === "function") cleanup();
await Promise.resolve();
assert.equal(unlistened, 6);
console.log(
  "RecordingOverlay: startup, readiness, stale settings and cleanup passed",
);
