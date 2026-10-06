/**
 * Request validation and response mapping for POST /api/recommendations.
 *
 * Public contract (camelCase):
 *   request:  { email: string, limit?: number, excludeProductIds?: string[] }
 *   response: { email, llmProvider, llmModel, llmUsed, recommendations: Recommendation[] }
 * The ML service uses snake_case; this module translates between the two.
 */

export interface RecommendationRequest {
  email: string;
  limit?: number;
  excludeProductIds: string[];
}

export interface Recommendation {
  rank: number;
  productid: string;
  title: string;
  category: string;
  price: number;
  imageurl: string | null;
  relevanceScore: number | null;
  modelScore: number;
  reason: string;
  source: "llm" | "model";
}

export interface RecommendationResponse {
  email: string;
  llmProvider: string;
  llmModel: string;
  llmUsed: boolean;
  recommendations: Recommendation[];
}

/** Shape returned by the Python ML service. */
interface MlRecommendation {
  rank: number;
  productid: string;
  title: string;
  category: string;
  price: number;
  imageurl: string | null;
  relevance_score: number | null;
  model_score: number;
  reason: string;
  source: "llm" | "model";
}

export interface MlRecommendationResponse {
  email: string;
  llm_provider: string;
  llm_model: string;
  llm_used: boolean;
  recommendations: MlRecommendation[];
}

export type ValidationResult =
  | { ok: true; value: RecommendationRequest }
  | { ok: false; error: string };

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
const MAX_EXCLUDED = 200;

export function validateRequest(body: unknown, maxLimit: number): ValidationResult {
  if (typeof body !== "object" || body === null || Array.isArray(body)) {
    return { ok: false, error: "Body must be a JSON object" };
  }
  const { email, limit, excludeProductIds } = body as Record<string, unknown>;

  if (typeof email !== "string" || email.length > 254 || !EMAIL_RE.test(email.trim())) {
    return { ok: false, error: "A valid member email is required" };
  }
  if (limit !== undefined && (!Number.isInteger(limit) || (limit as number) < 1 || (limit as number) > maxLimit)) {
    return { ok: false, error: `limit must be an integer between 1 and ${maxLimit}` };
  }
  if (
    excludeProductIds !== undefined &&
    (!Array.isArray(excludeProductIds) ||
      excludeProductIds.length > MAX_EXCLUDED ||
      !excludeProductIds.every((id) => typeof id === "string" && id.length > 0 && id.length <= 255))
  ) {
    return { ok: false, error: "excludeProductIds must be an array of product id strings" };
  }

  return {
    ok: true,
    value: {
      email: email.trim().toLowerCase(),
      limit: limit as number | undefined,
      excludeProductIds: (excludeProductIds as string[] | undefined) ?? [],
    },
  };
}

export function toMlRequest(req: RecommendationRequest): Record<string, unknown> {
  return {
    email: req.email,
    ...(req.limit !== undefined ? { limit: req.limit } : {}),
    exclude_product_ids: req.excludeProductIds,
  };
}

/** Map the ML response to the public shape and guarantee relevance order. */
export function fromMlResponse(ml: MlRecommendationResponse): RecommendationResponse {
  return {
    email: ml.email,
    llmProvider: ml.llm_provider,
    llmModel: ml.llm_model,
    llmUsed: ml.llm_used,
    recommendations: [...ml.recommendations]
      .sort((a, b) => a.rank - b.rank)
      .map((r) => ({
        rank: r.rank,
        productid: r.productid,
        title: r.title,
        category: r.category,
        price: r.price,
        imageurl: r.imageurl,
        relevanceScore: r.relevance_score,
        modelScore: r.model_score,
        reason: r.reason,
        source: r.source,
      })),
  };
}
