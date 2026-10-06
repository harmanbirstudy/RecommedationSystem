/**
 * Runtime configuration for the public API, read from environment variables.
 */

function intFromEnv(name: string, fallback: number): number {
  const parsed = Number.parseInt(process.env[name] ?? "", 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export const config = {
  /** Base URL of the Python ML service. */
  mlServiceUrl: (process.env.ML_SERVICE_URL ?? "http://localhost:8001").replace(/\/+$/, ""),
  /** Browser origins allowed to call the API (Angular dev server, Spring Boot-served build). */
  allowedOrigins: (process.env.ALLOWED_ORIGINS ?? "http://localhost:4200,http://localhost:8080")
    .split(",")
    .map((o) => o.trim())
    .filter(Boolean),
  /** Upper bound on `limit`; the default count is decided by the ML service. */
  maxRecommendationCount: intFromEnv("MAX_RECOMMENDATION_COUNT", 20),
  /** Generous: a local Ollama model on CPU can take a while. */
  upstreamTimeoutMs: intFromEnv("UPSTREAM_TIMEOUT_SECONDS", 120) * 1000,
};
