import type { ChatResult } from "../api/schemas";

export function sseBody(events: Array<[string, unknown]>): string {
  return events.map(([event, data]) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`).join("");
}

/** A fetch Response whose body arrives in small chunks, like a real stream. */
export function sseResponse(events: Array<[string, unknown]>, chunkSize = 17): Response {
  const text = sseBody(events);
  const encoder = new TextEncoder();

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      for (let index = 0; index < text.length; index += chunkSize) {
        controller.enqueue(encoder.encode(text.slice(index, index + chunkSize)));
      }
      controller.close();
    },
  });

  return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

export function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

export const routineResult: ChatResult = {
  thread_id: "med_123",
  answer: "Asthma attacks are triggered by allergens and smoke [PDF p.394].",
  spoken: "Allergens and smoke can trigger asthma attacks.",
  language: "en",
  urgency: "routine",
  in_scope: true,
  source_used: "private_kb",
  grounding: "supported",
  usefulness: "useful",
  triage_reason: "Educational question.",
  image_description: "",
  citations: [
    {
      id: "kb-394",
      type: "kb",
      title: "Gale Encyclopedia of Medicine (2002)",
      label: "PDF p.394",
      page: 394,
      url: "",
      score: 0.62,
      excerpt: "Asthma attacks are triggered by allergens, smoke and exercise.",
    },
  ],
  trace: [
    { node: "triage", label: "Emergency triage", detail: "routine · in scope · en" },
    { node: "retrieve", label: "Knowledge retrieval", detail: "5 passages" },
    { node: "grade_documents", label: "Relevance grading", detail: "1/5 relevant" },
    { node: "generate", label: "Answer generation", detail: "1 sources · en" },
    { node: "check_grounding", label: "Grounding check", detail: "supported" },
    { node: "check_usefulness", label: "Usefulness check", detail: "useful" },
    { node: "finalize", label: "Conversation saved", detail: "turn saved to memory" },
  ],
  research_calls: [],
};

export const researchResult: ChatResult = {
  ...routineResult,
  answer: "Biologics such as dupilumab reduce severe attacks [PMID:34655640; FDA: dupilumab].",
  source_used: "research",
  citations: [
    {
      id: "pubmed-34655640",
      type: "pubmed",
      title: "Advances in asthma (J Allergy Clin Immunol, 2021)",
      label: "PMID:34655640",
      page: null,
      url: "https://pubmed.ncbi.nlm.nih.gov/34655640/",
      score: null,
      excerpt: "Biologics reduce exacerbations.",
    },
  ],
  research_calls: [{ tool: "pubmed", query: "asthma biologics", results: 3 }],
};

export const emergencyResult: ChatResult = {
  ...routineResult,
  answer: "**This may be a medical emergency. Call your local emergency number now.**",
  urgency: "emergency",
  source_used: "emergency",
  grounding: "skipped",
  citations: [],
  trace: [
    { node: "triage", label: "Emergency triage", detail: "emergency · in scope · en" },
    { node: "emergency", label: "Emergency response", detail: "emergency services advised" },
    { node: "finalize", label: "Conversation saved", detail: "turn saved to memory" },
  ],
};

export function answerEvents(result: ChatResult): Array<[string, unknown]> {
  return [
    ["start", { thread_id: result.thread_id }],
    ...result.trace.map((step): [string, unknown] => ["step", step]),
    ["result", result],
  ];
}
