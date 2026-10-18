import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";
import { useChatStore } from "./store/chatStore";
import {
  answerEvents,
  emergencyResult,
  jsonResponse,
  researchResult,
  routineResult,
  sseResponse,
} from "./test/fixtures";

const initialState = useChatStore.getState();
const fetchMock = vi.fn();
// Responses for /api/chat/stream, in call order.
let chatResponses: Response[] = [];
let conversations: unknown[] = [];

beforeEach(() => {
  useChatStore.setState(initialState, true);
  chatResponses = [];
  conversations = [];
  fetchMock.mockReset();
  fetchMock.mockImplementation(async (url: string, init?: RequestInit) => {
    if (url.endsWith("/api/health")) {
      return jsonResponse(200, {
        status: "ok",
        version: "2.0.0",
        models: { llm: "gemini-3.5-flash-lite", embeddings: "llama-text-embed-v2" },
      });
    }
    if (url.endsWith("/api/conversations") && !init?.method) {
      return jsonResponse(200, { conversations });
    }
    if (url.endsWith("/api/chat/stream")) {
      return chatResponses.shift() ?? jsonResponse(500, {});
    }
    return jsonResponse(404, { success: false, error: "Not found", code: "not_found" });
  });
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => vi.unstubAllGlobals());

async function ask(question: string) {
  const user = userEvent.setup();
  render(<App />);
  await user.type(screen.getByLabelText("Describe your symptoms or question"), question);
  await user.click(screen.getByRole("button", { name: /Ask/ }));
  return user;
}

describe("App", () => {
  it("shows the API status and the models from the health endpoint", async () => {
    render(<App />);
    expect(await screen.findByText("API online")).toBeInTheDocument();
    expect(await screen.findByText("gemini-3.5-flash-lite")).toBeInTheDocument();
  });

  it("streams an answer with triage, grounding, citations and the agent trace", async () => {
    chatResponses.push(sseResponse(answerEvents(routineResult)));
    await ask("What triggers asthma?");

    const answer = await screen.findByRole("article", { name: "MedGuide answer" });
    expect(within(answer).getByText("Triage · routine")).toBeInTheDocument();
    expect(within(answer).getByText("Grounded in sources")).toBeInTheDocument();
    expect(within(answer).getByText("PDF p.394")).toBeInTheDocument();

    const inspector = screen.getByRole("complementary", { name: "Agent inspector" });
    expect(within(inspector).getByText("Relevance grading")).toBeInTheDocument();
    expect(within(inspector).getByText("1/5 relevant")).toBeInTheDocument();
    expect(within(inspector).getByText("score 0.62")).toBeInTheDocument();
    // The gateway reports which provider served each LLM step.
    expect(within(inspector).getByText("via groq:openai/gpt-oss-120b")).toBeInTheDocument();

    const body = JSON.parse(String(fetchMock.mock.calls.find(([u]) => String(u).endsWith("/api/chat/stream"))![1].body));
    expect(body).toEqual({ message: "What triggers asthma?", thread_id: null, image_data: null });
  });

  it("sends follow-up questions on the same thread", async () => {
    chatResponses.push(sseResponse(answerEvents(routineResult)));
    chatResponses.push(sseResponse(answerEvents(routineResult)));
    const user = await ask("What triggers asthma?");
    await screen.findByRole("article", { name: "MedGuide answer" });

    await user.type(screen.getByLabelText("Describe your symptoms or question"), "And in children?");
    await user.click(screen.getByRole("button", { name: /Ask/ }));
    await screen.findAllByRole("article", { name: "MedGuide answer" });

    const calls = fetchMock.mock.calls.filter(([u]) => String(u).endsWith("/api/chat/stream"));
    expect(JSON.parse(String(calls[1][1].body)).thread_id).toBe("med_123");
  });

  it("shows MCP tool calls and research citations in the inspector", async () => {
    chatResponses.push(sseResponse(answerEvents(researchResult)));
    await ask("Newest asthma biologic?");

    const answer = await screen.findByRole("article", { name: "MedGuide answer" });
    expect(within(answer).getByText("PMID:34655640")).toBeInTheDocument();
    expect(within(answer).getByText("FDA: dupilumab")).toBeInTheDocument();

    const inspector = screen.getByRole("complementary", { name: "Agent inspector" });
    expect(within(inspector).getByText("MCP tool calls")).toBeInTheDocument();
    expect(within(inspector).getByText("asthma biologics")).toBeInTheDocument();
    expect(within(inspector).getByText("3 results")).toBeInTheDocument();
    expect(within(inspector).getByText("Research tools (MCP)")).toBeInTheDocument();
  });

  it("highlights an emergency answer", async () => {
    chatResponses.push(sseResponse(answerEvents(emergencyResult)));
    await ask("My father has crushing chest pain");

    const answer = await screen.findByRole("article", { name: "MedGuide answer" });
    expect(answer).toHaveClass("emergency");
    expect(within(answer).getByText("Emergency")).toBeInTheDocument();
    expect(within(answer).getByText(/Call your local emergency number now/)).toBeInTheDocument();
  });

  it("shows a stream error in a dismissible banner", async () => {
    chatResponses.push(
      sseResponse([
        ["start", { thread_id: "med_1" }],
        ["error", { success: false, error: "The AI provider is rate-limiting requests.", code: "provider_rate_limited" }],
      ]),
    );
    const user = await ask("What triggers asthma?");

    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("The AI provider is rate-limiting requests.");
    await user.click(within(alert).getByRole("button", { name: "Dismiss error" }));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("starts from an example question", async () => {
    chatResponses.push(sseResponse(answerEvents(routineResult)));
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole("button", { name: "What can trigger asthma attacks?" }));

    expect(await screen.findByRole("article", { name: "MedGuide answer" })).toBeInTheDocument();
  });
});
