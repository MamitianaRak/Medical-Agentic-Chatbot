export interface SseMessage {
  event: string;
  data: string;
}

/**
 * Parse a Server-Sent Events body. `EventSource` only supports GET, and the
 * chat endpoint is POST, so the stream is read with fetch instead.
 */
export async function* parseSseStream(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<SseMessage> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n");
    let boundary = buffer.indexOf("\n\n");

    while (boundary !== -1) {
      const message = parseBlock(buffer.slice(0, boundary));
      buffer = buffer.slice(boundary + 2);
      if (message) yield message;
      boundary = buffer.indexOf("\n\n");
    }
  }

  const last = parseBlock(buffer);
  if (last) yield last;
}

function parseBlock(block: string): SseMessage | null {
  let event = "message";
  const data: string[] = [];

  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
  }

  return data.length ? { event, data: data.join("\n") } : null;
}
