import { BookOpenCheck, ShieldCheck, Siren, Stethoscope } from "lucide-react";

const EXAMPLES = [
  "What can trigger asthma attacks?",
  "Quels sont les symptômes d'une anémie ?",
  "When is a fever in an adult a reason to see a doctor?",
];

const SAFEGUARDS = [
  {
    icon: Siren,
    title: "Emergency triage",
    text: "Red-flag symptoms get immediate emergency guidance.",
  },
  {
    icon: BookOpenCheck,
    title: "Cited sources",
    text: "Answers come from a medical encyclopedia or trusted web pages.",
  },
  {
    icon: ShieldCheck,
    title: "Checked answers",
    text: "Each answer is verified against its sources before you see it.",
  },
];

interface Props {
  onExample: (question: string) => void;
}

export function EmptyState({ onExample }: Props) {
  return (
    <div className="empty-state">
      <span className="empty-mark" aria-hidden>
        <Stethoscope size={28} />
      </span>
      <h1>How can I help you today?</h1>
      <p className="muted">
        Describe a symptom or ask about a condition, by text, voice or photo.
      </p>

      <ul className="safeguards">
        {SAFEGUARDS.map(({ icon: Icon, title, text }) => (
          <li key={title}>
            <span className="safeguard-icon" aria-hidden>
              <Icon size={18} />
            </span>
            <strong>{title}</strong>
            <span className="muted small">{text}</span>
          </li>
        ))}
      </ul>

      <div className="examples">
        <span className="eyebrow">Try a question</span>
        {EXAMPLES.map((example) => (
          <button key={example} type="button" className="chip" onClick={() => onExample(example)}>
            {example}
          </button>
        ))}
      </div>
    </div>
  );
}
