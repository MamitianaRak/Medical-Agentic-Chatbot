import { z } from "zod";

// Runtime validation of every API payload: a backend contract change fails
// loudly here instead of rendering a broken UI.

export const citationSchema = z.object({
  id: z.string(),
  type: z.enum(["kb", "pubmed", "fda", "medlineplus", "web"]),
  title: z.string(),
  label: z.string(),
  page: z.number().nullable().default(null),
  url: z.string().default(""),
  score: z.number().nullable().default(null),
  excerpt: z.string(),
});

export const toolCallSchema = z.object({
  tool: z.string(),
  query: z.string(),
  results: z.number(),
});

export const traceStepSchema = z.object({
  node: z.string(),
  label: z.string(),
  detail: z.string().default(""),
  /** LLM steps: "<provider>:<model>" that answered through the gateway. */
  model: z.string().default(""),
});

export const chatResultSchema = z.object({
  thread_id: z.string(),
  answer: z.string(),
  spoken: z.string().default(""),
  language: z.string().default("en"),
  urgency: z.enum(["emergency", "urgent", "routine"]),
  in_scope: z.boolean().default(true),
  source_used: z.string(),
  grounding: z.enum(["supported", "partial", "unsupported", "skipped"]),
  usefulness: z.enum(["useful", "not_useful", "skipped"]).default("skipped"),
  triage_reason: z.string().default(""),
  image_description: z.string().default(""),
  citations: z.array(citationSchema).default([]),
  research_calls: z.array(toolCallSchema).default([]),
  trace: z.array(traceStepSchema).default([]),
});

export const startEventSchema = z.object({ thread_id: z.string() });

export const conversationSummarySchema = z.object({
  thread_id: z.string(),
  title: z.string(),
  turns: z.number(),
  created_at: z.string(),
  updated_at: z.string(),
});

export const conversationListSchema = z.object({
  conversations: z.array(conversationSummarySchema),
});

export const conversationDetailSchema = conversationSummarySchema.extend({
  messages: z.array(z.object({ question: z.string(), answer: z.string() })),
});

export const healthSchema = z.object({
  status: z.string(),
  version: z.string(),
  models: z.record(z.string(), z.string()).default({}),
});

export const apiErrorSchema = z.object({
  success: z.literal(false),
  error: z.string(),
  code: z.string(),
});

export type Citation = z.infer<typeof citationSchema>;
export type TraceStep = z.infer<typeof traceStepSchema>;
export type ToolCall = z.infer<typeof toolCallSchema>;
export type ChatResult = z.infer<typeof chatResultSchema>;
export type ConversationSummary = z.infer<typeof conversationSummarySchema>;
export type ConversationDetail = z.infer<typeof conversationDetailSchema>;
export type Health = z.infer<typeof healthSchema>;
