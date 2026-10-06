import { describe, expect, it } from "vitest";
import { fromMlResponse, toMlRequest, validateRequest } from "./recommendations";

describe("validateRequest", () => {
  it("accepts an email and normalises it", () => {
    const result = validateRequest({ email: "  Chris.Walker@Example.com " }, 20);
    expect(result).toEqual({
      ok: true,
      value: { email: "chris.walker@example.com", limit: undefined, excludeProductIds: [] },
    });
  });

  it.each([
    [null, "Body must be a JSON object"],
    [{}, "A valid member email is required"],
    [{ email: "not-an-email" }, "A valid member email is required"],
    [{ email: "a@b.co", limit: 0 }, "limit must be an integer between 1 and 20"],
    [{ email: "a@b.co", limit: 21 }, "limit must be an integer between 1 and 20"],
    [{ email: "a@b.co", limit: 2.5 }, "limit must be an integer between 1 and 20"],
    [{ email: "a@b.co", excludeProductIds: "p1" }, "excludeProductIds must be an array of product id strings"],
    [{ email: "a@b.co", excludeProductIds: [1] }, "excludeProductIds must be an array of product id strings"],
  ])("rejects %j", (body, error) => {
    expect(validateRequest(body, 20)).toEqual({ ok: false, error });
  });
});

describe("toMlRequest", () => {
  it("omits limit when not given and uses snake_case", () => {
    expect(toMlRequest({ email: "a@b.co", excludeProductIds: ["p1"] })).toEqual({
      email: "a@b.co",
      exclude_product_ids: ["p1"],
    });
  });
});

describe("fromMlResponse", () => {
  it("maps to camelCase and orders by rank", () => {
    const rec = (rank: number, productid: string) => ({
      rank,
      productid,
      title: productid,
      category: "Books",
      price: 10,
      imageurl: null,
      relevance_score: 100 - rank,
      model_score: 0.5,
      reason: "r",
      source: "llm" as const,
    });
    const out = fromMlResponse({
      email: "a@b.co",
      llm_provider: "ollama",
      llm_model: "qwen2.5-coder:7b",
      llm_used: true,
      recommendations: [rec(2, "b"), rec(1, "a")],
    });
    expect(out.llmProvider).toBe("ollama");
    expect(out.recommendations.map((r) => r.productid)).toEqual(["a", "b"]);
    expect(out.recommendations[0].relevanceScore).toBe(99);
  });
});
