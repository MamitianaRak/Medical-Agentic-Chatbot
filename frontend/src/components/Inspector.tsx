import { BookOpen, Check, ExternalLink, FileText, Globe, Microscope, Pill } from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { Fragment } from "react";

import type { ChatResult, TraceStep } from "../api/schemas";
import { CITATION_SOURCE, SOURCE_LABEL, TOOL_LABEL, formatScore } from "../lib/format";
import { DrawerHeader } from "./DrawerHeader";

interface Props {
  result: ChatResult | null;
  liveSteps: TraceStep[];
  thinking: boolean;
  models: Record<string, string>;
  onClose: () => void;
}

const MODEL_LABELS: Array<[string, string]> = [
  ["llm", "LLM · answer"],
  ["llm_fast", "LLM · fast steps"],
  ["embeddings", "Embeddings"],
  ["vision", "Vision"],
  ["speech_to_text", "Speech-to-text"],
  ["text_to_speech", "Text-to-speech"],
  ["research_tools", "Research tools"],
];

const SOURCE_ICON: Record<string, LucideIcon> = {
  kb: FileText,
  pubmed: Microscope,
  fda: Pill,
  medlineplus: BookOpen,
  web: Globe,
};

function Trace({ steps, running }: { steps: TraceStep[]; running: boolean }) {
  return (
    <ol className="trace">
      {steps.map((step, index) => (
        <li key={`${step.node}-${index}`} className="trace-step">
          <span className="trace-icon done" aria-hidden>
            <Check size={12} strokeWidth={3} />
          </span>
          <span className="trace-label">{step.label}</span>
          <span className="trace-detail mono">{step.detail}</span>
          {step.model ? <span className="trace-model mono">via {step.model}</span> : null}
        </li>
      ))}
      {running ? (
        <li className="trace-step">
          <span className="trace-icon" aria-hidden>
            <span className="spinner spinner-sm" />
          </span>
          <span className="trace-label muted">Running…</span>
        </li>
      ) : null}
    </ol>
  );
}

export function Inspector({ result, liveSteps, thinking, models, onClose }: Props) {
  const steps = thinking ? liveSteps : (result?.trace ?? []);
  const modelRows = MODEL_LABELS.filter(([key]) => models[key]);

  return (
    <aside className="inspector" aria-label="Agent inspector">
      <DrawerHeader title="Agent inspector" onClose={onClose} />
      <section>
        <h2 className="eyebrow">Agent trace</h2>
        {steps.length || thinking ? (
          <Trace steps={steps} running={thinking} />
        ) : (
          <p className="muted small">Ask a question to see each step of the LangGraph workflow.</p>
        )}
      </section>

      {result && !thinking ? (
        <>
          <section>
            <h2 className="eyebrow">Run</h2>
            <dl className="run-meta mono">
              <dt>source</dt>
              <dd>{SOURCE_LABEL[result.source_used] ?? result.source_used}</dd>
              <dt>grounding</dt>
              <dd>{result.grounding}</dd>
              <dt>usefulness</dt>
              <dd>{result.usefulness.replace("_", " ")}</dd>
              <dt>language</dt>
              <dd>{result.language}</dd>
              {result.image_description ? (
                <>
                  <dt>image</dt>
                  <dd>{result.image_description}</dd>
                </>
              ) : null}
            </dl>
          </section>

          {result.research_calls.length ? (
            <section>
              <h2 className="eyebrow">MCP tool calls</h2>
              <ul className="tool-calls">
                {result.research_calls.map((call, index) => (
                  <li key={`${call.tool}-${index}`}>
                    <span className="tool-name">{TOOL_LABEL[call.tool] ?? call.tool}</span>
                    <span className="mono small muted">{call.query}</span>
                    <span className="mono small">
                      {call.results} result{call.results === 1 ? "" : "s"}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}

          <section>
            <h2 className="eyebrow">Sources</h2>
            {result.citations.length === 0 ? (
              <p className="muted small">No sources were used for this answer.</p>
            ) : (
              <ul className="sources">
                {result.citations.map((citation) => {
                  const Icon = SOURCE_ICON[citation.type] ?? Globe;
                  return (
                    <li key={citation.id} className="source-card">
                      <div className="source-head mono">
                        <span className="source-ref">
                          <Icon size={13} aria-hidden />
                          {citation.label}
                        </span>
                        <span>
                          {citation.score !== null
                            ? `score ${formatScore(citation.score)}`
                            : CITATION_SOURCE[citation.type]}
                        </span>
                      </div>
                      {citation.score !== null ? (
                        <div
                          className="score-bar"
                          role="meter"
                          aria-label="Relevance score"
                          aria-valuemin={0}
                          aria-valuemax={1}
                          aria-valuenow={citation.score}
                        >
                          <span style={{ width: `${Math.min(citation.score, 1) * 100}%` }} />
                        </div>
                      ) : null}
                      {citation.url ? (
                        <a href={citation.url} target="_blank" rel="noreferrer noopener">
                          {citation.title} <ExternalLink size={12} aria-hidden />
                        </a>
                      ) : (
                        <strong className="small">{citation.title}</strong>
                      )}
                      <p className="small muted">{citation.excerpt}</p>
                    </li>
                  );
                })}
              </ul>
            )}
          </section>
        </>
      ) : null}

      {modelRows.length ? (
        <section>
          <h2 className="eyebrow">Models</h2>
          <dl className="run-meta mono">
            {modelRows.map(([key, label]) => (
              <Fragment key={key}>
                <dt>{label}</dt>
                <dd>{models[key]}</dd>
              </Fragment>
            ))}
          </dl>
        </section>
      ) : null}
    </aside>
  );
}
