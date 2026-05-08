/** Shared constants for the Repo-Aware AI extension. */

/** Default backend server port. Can be overridden via settings. */
export const DEFAULT_PORT = 8360;

/** Backend poll/health check interval in ms during startup */
export const HEALTH_POLL_INTERVAL_MS = 500;

/** Maximum wait time for backend startup (30 seconds) */
export const BACKEND_STARTUP_TIMEOUT_MS = 30_000;

/** Status poll interval when idle (5 seconds) */
export const STATUS_POLL_IDLE_MS = 5_000;

/** Status poll interval while indexing (500ms) */
export const STATUS_POLL_BUSY_MS = 500;

/** Debounce delay after file save before re-indexing */
export const AUTO_INDEX_DEBOUNCE_MS = 5_000;

/** Hover provider timeout — hide if backend is slow */
export const HOVER_TIMEOUT_MS = 2_000;

/** LRU hover cache capacity */
export const HOVER_CACHE_SIZE = 100;

/** CodeLens debounce delay after document change */
export const CODELENS_DEBOUNCE_MS = 2_000;

/** Max symbol count per file for CodeLens */
export const CODELENS_MAX_SYMBOLS = 50;

/** Output channel name */
export const OUTPUT_CHANNEL_NAME = "Repo AI Backend";

/** Extension ID */
export const EXTENSION_ID = "repo-aware-ai";

/** Backend health endpoint path */
export const HEALTH_PATH = "/health";

/** Configuration section key */
export const CONFIG_SECTION = "repoAwareAI";
