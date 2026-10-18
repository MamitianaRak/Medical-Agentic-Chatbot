import { X } from "lucide-react";

interface Props {
  title: string;
  onClose: () => void;
}

/** Title and close button shown when a side panel is displayed as a drawer (small screens). */
export function DrawerHeader({ title, onClose }: Props) {
  return (
    <div className="drawer-head">
      <strong>{title}</strong>
      <button type="button" className="icon-btn icon-btn-lg" onClick={onClose} aria-label={`Close ${title.toLowerCase()}`}>
        <X size={18} />
      </button>
    </div>
  );
}
