import {
  apiErrorSchema,
  chatResultSchema,
  conversationDetailSchema,
  conversationListSchema,
  healthSchema,
  startEventSchema,
  traceStepSchema,
} from "./schemas";
import type { ChatResult, ConversationDetail, ConversationSummary, Health, TraceStep } from "./schemas";
import { parseSseStream } from "./sse";

// Empty in development (Vite proxy) and with nginx; the backend URL on Render.
export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status?: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function readError(response: Response): Promise<ApiError> {
  const body = await response.json().catch(() => null);
  const parsed = apiErrorSchema.safeParse(body);

  if (parsed.success) {
    return new ApiError(parsed.data.error, parsed.data.code, response.status);
  }
  return new ApiError(`Request failed (${response.status}).`, "http_error", response.status);
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) throw await readError(response);
  return response;
}

export async function getHealth(signal?: AbortSignal): Promise<Health | null> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`, { signal });
    return response.ok ? healthSchema.parse(await response.json()) : null;
  } catch {
    return null;
  }
}

export async function listConversations(): Promise<ConversationSummary[]> {
  const response = await request("/api/conversations");
  return conversationListSchema.parse(await response.json()).conversations;
}

export async function getConversation(threadId: string): Promise<ConversationDetail> {
  const response = await request(`/api/conversations/${encodeURIComponent(threadId)}`);
  return conversationDetailSchema.parse(await response.json());
}

export async function deleteConversation(threadId: string): Promise<void> {
  await request(`/api/conversations/${encodeURIComponent(threadId)}`, { method: "DELETE" });
}

export interface ChatPayload {
  message: string;
  thread_id?: string | null;
  image_data?: string | null;
}

export interface StreamHandlers {
  onStart?: (threadId: string) => void;
  onStep: (step: TraceStep) => void;
}

/** Ask a question and follow the workflow live, one `step` event per node. */
export async function streamChat(
  payload: ChatPayload,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<ChatResult> {
  const response = await fetch(`${API_BASE_URL}/api/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!response.ok || !response.body) {
    throw await readError(response);
  }

  let result: ChatResult | null = null;

  for await (const { event, data } of parseSseStream(response.body)) {
    const json: unknown = JSON.parse(data);

    if (event === "start") {
      handlers.onStart?.(startEventSchema.parse(json).thread_id);
    } else if (event === "step") {
      handlers.onStep(traceStepSchema.parse(json));
    } else if (event === "result") {
      result = chatResultSchema.parse(json);
    } else if (event === "error") {
      const error = apiErrorSchema.parse(json);
      throw new ApiError(error.error, error.code);
    }
  }

  if (!result) {
    throw new ApiError("The server closed the stream without an answer.", "stream_incomplete");
  }
  return result;
}

export async function transcribe(audio: Blob): Promise<string> {
  const form = new FormData();
  form.append("file", audio, "question.webm");
  const response = await request("/api/speech/transcribe", { method: "POST", body: form });
  const body = (await response.json()) as { text?: string };
  return body.text ?? "";
}

export async function synthesize(text: string, language: string): Promise<Blob> {
  const response = await request("/api/speech/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, language }),
  });
  return response.blob();
}
