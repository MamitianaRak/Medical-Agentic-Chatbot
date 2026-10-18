import { describe, expect, it } from "vitest";

import { sseResponse } from "../test/fixtures";
import { parseSseStream } from "./sse";

async function collect(body: ReadableStream<Uint8Array>) {
  const messages = [];
  for await (const message of parseSseStream(body)) messages.push(message);
  return messages;
}

describe("parseSseStream", () => {
  it("reassembles events split across chunks", async () => {
    const response = sseResponse(
      [
        ["step", { node: "triage" }],
        ["result", { ok: true }],
      ],
      5,
    );

    expect(await collect(response.body!)).toEqual([
      { event: "step", data: '{"node":"triage"}' },
      { event: "result", data: '{"ok":true}' },
    ]);
  });

  it("handles CRLF line endings and a final event without a trailing blank line", async () => {
    const body = new Response("event: step\r\ndata: 1\r\n\r\nevent: result\ndata: 2").body!;

    expect(await collect(body)).toEqual([
      { event: "step", data: "1" },
      { event: "result", data: "2" },
    ]);
  });
});
