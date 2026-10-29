import { Activity, Menu, Moon, Phone, Plus, Stethoscope, Sun } from "lucide-react";

import type { Theme } from "../lib/theme";
import type { ApiStatus } from "../lib/useHealth";

const STATUS_LABEL: Record<ApiStatus, string> = {
  checking: "Connecting",
  online: "API online",
  offline: "API offline",
};

interface Props {
  status: ApiStatus;
  theme: Theme;
  onToggleTheme: () => void;
  onNewConsultation: () => void;
  onOpenConversations: () => void;
  onOpenInspector: () => void;
  busy: boolean;
}

export function TopBar({
  status,
  theme,
  onToggleTheme,
  onNewConsultation,
  onOpenConversations,
  onOpenInspector,
  busy,
}: Props) {
  return (
    <>
      <div className="emergency-strip" role="note">
        <Phone size={14} aria-hidden />
        <span className="strip-long">
          <strong>Medical emergency?</strong> Call <strong>15</strong> (SAMU, France) ·{" "}
          <strong>112</strong> (Europe) · <strong>911</strong> (US). Do not wait for an online
          answer.
        </span>
        <span className="strip-short">
          <strong>Emergency?</strong> Call <strong>15</strong> · <strong>112</strong>
        </span>
      </div>

      <header className="topbar">
        <div className="brand">
          <button
            type="button"
            className="icon-btn icon-btn-lg menu-toggle"
            onClick={onOpenConversations}
            aria-label="Open consultations"
          >
            <Menu size={18} />
          </button>
          <span className="brand-mark" aria-hidden>
            <Stethoscope size={20} strokeWidth={2.2} />
          </span>
          <span className="brand-text">
            <span className="brand-name">MedGuide AI</span>
            <span className="brand-tag">Clinical knowledge assistant</span>
          </span>
        </div>
        <div className="topbar-actions">
          <span className={`api-status ${status}`} role="status">
            <span className="api-dot" aria-hidden />
            {STATUS_LABEL[status]}
          </span>
          <button
            type="button"
            className="icon-btn icon-btn-lg inspector-toggle"
            onClick={onOpenInspector}
            aria-label="Open agent trace"
          >
            <Activity size={18} />
          </button>
          <button
            type="button"
            className="icon-btn icon-btn-lg"
            onClick={onToggleTheme}
            aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
          >
            {theme === "light" ? <Moon size={18} /> : <Sun size={18} />}
          </button>
          <button
            type="button"
            className="btn btn-primary"
            onClick={onNewConsultation}
            disabled={busy}
            aria-label="New consultation"
          >
            <Plus size={16} aria-hidden />
            <span className="btn-label">New consultation</span>
          </button>
        </div>
      </header>
    </>
  );
}
