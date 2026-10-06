/**
 * POST /api/recommendations
 *
 * Body: { "email": "member@example.com", "limit"?: 5, "excludeProductIds"?: ["..."] }
 * Returns the member's recommended products, sorted by relevance (rank 1 first).
 *
 * Validates input, then proxies to the Python ML service (XGBoost + LLM rerank).
 * Errors: 400 invalid body, 404 unknown member, 503 ML service not ready,
 *         502 ML service error, 504 ML service timeout.
 */
import { config } from "@/lib/config";
import { corsHeaders, preflight } from "@/lib/cors";
import {
  fromMlResponse,
  toMlRequest,
  validateRequest,
  type MlRecommendationResponse,
} from "@/lib/recommendations";

export const dynamic = "force-dynamic";

function json(request: Request, body: unknown, status = 200): Response {
  return Response.json(body, {
    status,
    headers: { ...corsHeaders(request), "Cache-Control": "no-store" },
  });
}

export function OPTIONS(request: Request): Response {
  return preflight(request);
}

export async function POST(request: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return json(request, { error: "Body must be valid JSON" }, 400);
  }

  const validation = validateRequest(body, config.maxRecommendationCount);
  if (!validation.ok) {
    return json(request, { error: validation.error }, 400);
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${config.mlServiceUrl}/recommendations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(toMlRequest(validation.value)),
      signal: AbortSignal.timeout(config.upstreamTimeoutMs),
      cache: "no-store",
    });
  } catch (err) {
    const timedOut = err instanceof DOMException && err.name === "TimeoutError";
    console.error("ML service request failed:", err);
    return json(
      request,
      { error: timedOut ? "Recommendation service timed out" : "Recommendation service unreachable" },
      timedOut ? 504 : 502,
    );
  }

  if (upstream.status === 404) {
    return json(request, { error: "Member not found" }, 404);
  }
  if (upstream.status === 503) {
    return json(request, { error: "Recommendation service is not ready" }, 503);
  }
  if (!upstream.ok) {
    console.error("ML service returned", upstream.status, await upstream.text());
    return json(request, { error: "Recommendation service error" }, 502);
  }

  const ml = (await upstream.json()) as MlRecommendationResponse;
  return json(request, fromMlResponse(ml));
}
