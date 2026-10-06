/**
 * GET /api/health — liveness of this API plus readiness of the ML service.
 */
import { config } from "@/lib/config";

export const dynamic = "force-dynamic";

export async function GET(): Promise<Response> {
  try {
    const res = await fetch(`${config.mlServiceUrl}/health`, {
      signal: AbortSignal.timeout(5000),
      cache: "no-store",
    });
    const ml = await res.json();
    return Response.json({ status: "ok", mlService: ml }, { status: ml.ready ? 200 : 503 });
  } catch {
    return Response.json({ status: "ok", mlService: null }, { status: 503 });
  }
}
