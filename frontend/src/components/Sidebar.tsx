import { BookOpen, Trash2 } from "lucide-react";

import type { ConversationSummary } from "../api/schemas";

interface Props {
  conversations: ConversationSummary[];
  activeId: string | null;
  disabled: boolean;
  onOpen: (threadId: string) => void;
  onDelete: (threadId: string) => void;
}

export function Sidebar({ conversations, activeId, disabled, onOpen, onDelete }: Props) {
  return (
    <aside className="sidebar" aria-label="Consultations">
      <h2 className="eyebrow">Consultations</h2>

      {conversations.length === 0 ? (
        <p className="muted small">Your consultations will appear here.</p>
      ) : (
        <ul className="conversation-list">
          {conversations.map((conversation) => (
            <li
              key={conversation.thread_id}
              className={conversation.thread_id === activeId ? "active" : undefined}
            >
              <button
                type="button"
                className="conversation-open"
                onClick={() => onOpen(conversation.thread_id)}
                disabled={disabled}
                aria-current={conversation.thread_id === activeId ? "true" : undefined}
              >
                {conversation.title}
              </button>
              <button
                type="button"
                className="icon-btn"
                onClick={() => onDelete(conversation.thread_id)}
                disabled={disabled}
                aria-label={`Delete consultation: ${conversation.title}`}
              >
                <Trash2 size={15} />
              </button>
            </li>
          ))}
        </ul>
      )}

      <div className="kb-card">
        <span className="eyebrow">
          <BookOpen size={13} aria-hidden /> Knowledge base
        </span>
        <strong>Gale Encyclopedia of Medicine</strong>
        <span className="mono muted small">2002 · vol. 1 A–B · historical</span>
      </div>
    </aside>
  );
}
