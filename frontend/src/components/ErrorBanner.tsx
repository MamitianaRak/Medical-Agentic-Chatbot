import { AlertTriangle, X } from "lucide-react";

interface Props {
  message: string;
  onDismiss: () => void;
}

export function ErrorBanner({ message, onDismiss }: Props) {
  return (
    <div className="error-banner" role="alert">
      <AlertTriangle size={18} aria-hidden />
      <span>{message}</span>
      <button type="button" className="icon-btn" onClick={onDismiss} aria-label="Dismiss error">
        <X size={16} />
      </button>
    </div>
  );
}
