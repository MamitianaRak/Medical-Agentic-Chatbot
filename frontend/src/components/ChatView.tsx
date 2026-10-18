import { Stethoscope } from "lucide-react";
import { useEffect, useRef } from "react";

import type { TraceStep } from "../api/schemas";
import type { ChatMessage } from "../store/chatStore";
import { EmptyState } from "./EmptyState";
import { MessageItem } from "./MessageItem";

interface Props {
  messages: ChatMessage[];
  thinking: boolean;
  liveSteps: TraceStep[];
  selectedId: string | null;
  onSelect: (messageId: string) => void;
  onExample: (question: string) => void;
  onError: (message: string) => void;
}

export function ChatView({
  messages,
  thinking,
  liveSteps,
  selectedId,
  onSelect,
  onExample,
  onError,
}: Props) {
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    end.current?.scrollIntoView?.({ behavior: "smooth", block: "end" });
  }, [messages.length, thinking, liveSteps.length]);

  if (messages.length === 0 && !thinking) {
    return (
      <div className="chat-scroll">
        <EmptyState onExample={onExample} />
      </div>
    );
  }

  const current = liveSteps.at(-1);

  return (
    <div className="chat-scroll" aria-live="polite">
      <div className="chat-thread">
        {messages.map((message) => (
          <MessageItem
            key={message.id}
            message={message}
            selected={message.id === selectedId}
            onSelect={() => onSelect(message.id)}
            onError={onError}
          />
        ))}
        {thinking ? (
          <div className="msg-thinking" role="status">
            <span className="avatar" aria-hidden>
              <Stethoscope size={16} />
            </span>
            <span className="spinner" aria-hidden />
            <span>
              {current ? `Working… last step: ${current.label}` : "Starting the workflow…"}
            </span>
          </div>
        ) : null}
        <div ref={end} />
      </div>
    </div>
  );
}
