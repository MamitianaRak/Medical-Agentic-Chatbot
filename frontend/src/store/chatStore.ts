import { create } from "zustand";

import {
  ApiError,
  deleteConversation,
  getConversation,
  listConversations,
  streamChat,
} from "../api/client";
import type { ChatResult, ConversationSummary, TraceStep } from "../api/schemas";

export interface UserMessage {
  id: string;
  role: "user";
  text: string;
  image?: string | null;
}

export interface AssistantMessage {
  id: string;
  role: "assistant";
  text: string;
  /** Full workflow output; absent for messages reloaded from history. */
  result?: ChatResult;
}

export type ChatMessage = UserMessage | AssistantMessage;

interface ChatState {
  threadId: string | null;
  messages: ChatMessage[];
  conversations: ConversationSummary[];
  thinking: boolean;
  liveSteps: TraceStep[];
  selectedId: string | null;
  error: string | null;

  refreshConversations: () => Promise<void>;
  openConversation: (threadId: string) => Promise<void>;
  newConversation: () => void;
  removeConversation: (threadId: string) => Promise<void>;
  send: (text: string, image?: string | null) => Promise<void>;
  select: (messageId: string) => void;
  setError: (message: string | null) => void;
}

let counter = 0;
const nextId = (prefix: string) => `${prefix}-${++counter}`;

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  return "Could not reach MedGuide AI. Check your connection and try again.";
}

export const useChatStore = create<ChatState>((set, get) => ({
  threadId: null,
  messages: [],
  conversations: [],
  thinking: false,
  liveSteps: [],
  selectedId: null,
  error: null,

  refreshConversations: async () => {
    try {
      set({ conversations: await listConversations() });
    } catch (error) {
      set({ error: errorMessage(error) });
    }
  },

  openConversation: async (threadId) => {
    if (get().thinking) return;
    try {
      const detail = await getConversation(threadId);
      const messages: ChatMessage[] = detail.messages.flatMap((turn) => [
        { id: nextId("u"), role: "user" as const, text: turn.question },
        { id: nextId("a"), role: "assistant" as const, text: turn.answer },
      ]);
      set({ threadId, messages, liveSteps: [], selectedId: null, error: null });
    } catch (error) {
      set({ error: errorMessage(error) });
    }
  },

  newConversation: () => {
    if (get().thinking) return;
    set({ threadId: null, messages: [], liveSteps: [], selectedId: null, error: null });
  },

  removeConversation: async (threadId) => {
    try {
      await deleteConversation(threadId);
      if (get().threadId === threadId) get().newConversation();
      await get().refreshConversations();
    } catch (error) {
      set({ error: errorMessage(error) });
    }
  },

  send: async (text, image = null) => {
    const message = text.trim();
    if (!message || get().thinking) return;

    const userMessage: UserMessage = { id: nextId("u"), role: "user", text: message, image };
    set((state) => ({
      messages: [...state.messages, userMessage],
      thinking: true,
      liveSteps: [],
      selectedId: null,
      error: null,
    }));

    try {
      const result = await streamChat(
        { message, thread_id: get().threadId, image_data: image },
        {
          onStart: (threadId) => set({ threadId }),
          onStep: (step) => set((state) => ({ liveSteps: [...state.liveSteps, step] })),
        },
      );
      const answer: AssistantMessage = {
        id: nextId("a"),
        role: "assistant",
        text: result.answer,
        result,
      };
      set((state) => ({
        messages: [...state.messages, answer],
        threadId: result.thread_id,
        thinking: false,
        selectedId: answer.id,
      }));
      void get().refreshConversations();
    } catch (error) {
      set({ thinking: false, error: errorMessage(error) });
    }
  },

  select: (messageId) => set({ selectedId: messageId }),
  setError: (message) => set({ error: message }),
}));

/** The assistant message whose workflow the inspector shows. */
export function selectedResult(state: ChatState): ChatResult | null {
  const message = state.messages.find((m) => m.id === state.selectedId);
  return message?.role === "assistant" ? (message.result ?? null) : null;
}
