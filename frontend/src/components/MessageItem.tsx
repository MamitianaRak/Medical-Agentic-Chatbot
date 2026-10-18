import { Activity, AlertTriangle, PanelRight, ShieldCheck, Stethoscope } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { GROUNDING_LABEL, URGENCY_LABEL, withCitationChips } from "../lib/format";
import type { ChatMessage } from "../store/chatStore";
import { ListenButton } from "./ListenButton";

interface Props {
  message: ChatMessage;
  selected: boolean;
  onSelect: () => void;
  onError: (message: string) => void;
}

function Markdown({ text }: { text: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noreferrer noopener">
              {children}
            </a>
          ),
        }}
      >
        {withCitationChips(text)}
      </ReactMarkdown>
    </div>
  );
}

export function MessageItem({ message, selected, onSelect, onError }: Props) {
  if (message.role === "user") {
    return (
      <div className="msg-user">
        {message.image ? <img src={message.image} alt="Attached by you" className="msg-image" /> : null}
        <p>{message.text}</p>
      </div>
    );
  }

  const result = message.result;
  const emergency = result?.urgency === "emergency";
  const grounding = result ? GROUNDING_LABEL[result.grounding] : "";
  const historical = result?.citations.some((c) => c.type === "kb");

  return (
    <div className="msg-row">
      <span className={`avatar${emergency ? " avatar-emergency" : ""}`} aria-hidden>
        <Stethoscope size={16} />
      </span>
      <article
        className={`msg-assistant${emergency ? " emergency" : ""}${selected ? " selected" : ""}`}
        aria-label="MedGuide answer"
      >
        {result ? (
          <div className="badges">
            <span className={`badge badge-${result.urgency}`}>
              {emergency ? (
                <AlertTriangle size={13} aria-hidden />
              ) : (
                <Activity size={13} aria-hidden />
              )}
              {URGENCY_LABEL[result.urgency]}
            </span>
            {grounding ? (
              <span className={`badge badge-grounding-${result.grounding}`}>
                <ShieldCheck size={13} aria-hidden />
                {grounding}
              </span>
            ) : null}
          </div>
        ) : null}

        <Markdown text={message.text} />

        {result ? (
          <footer className="msg-footer">
            <span className="muted small">
              {historical
                ? "Main reference dates from 2002. Confirm with current guidance."
                : "Educational information only. Not a diagnosis."}
            </span>
            <div className="msg-actions">
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={onSelect}
                aria-pressed={selected}
              >
                <PanelRight size={15} aria-hidden />
                Trace
              </button>
              <ListenButton
                text={result.spoken || result.answer}
                language={result.language}
                onError={onError}
              />
            </div>
          </footer>
        ) : null}
      </article>
    </div>
  );
}
