import type { ChatResult } from "../api/schemas";

// One citation label: PDF p.394 · PMID:31678040 · FDA: ibuprofen · MedlinePlus: Asthma · Web: who.int
const LABEL = String.raw`(?:PDF p\.\d+|PMID:\d+|FDA: [^\],;]+|MedlinePlus: [^\],;]+|Web: [^\],;\s]+)`;
const CITATION = new RegExp(String.raw`\[(${LABEL}(?:[,;]\s*${LABEL})*)\]`, "g");

/**
 * Turn citation labels into inline code so they render as compact chips:
 * `[PDF p.394]` and `[PDF p.250; PMID:123]` become separate `PDF p.394` spans.
 */
export function withCitationChips(markdown: string): string {
  return markdown.replace(CITATION, (_, labels: string) =>
    labels
      .split(/[,;]\s*(?=PDF|PMID|FDA|MedlinePlus|Web)/)
      .map((label) => `\`${label.trim()}\``)
      .join(" "),
  );
}

export const URGENCY_LABEL: Record<ChatResult["urgency"], string> = {
  emergency: "Emergency",
  urgent: "Urgent · see a clinician",
  routine: "Triage · routine",
};

export const GROUNDING_LABEL: Record<ChatResult["grounding"], string> = {
  supported: "Grounded in sources",
  partial: "Partially grounded",
  unsupported: "Not verified",
  skipped: "",
};

export const SOURCE_LABEL: Record<string, string> = {
  private_kb: "Medical textbook",
  research: "Research tools (MCP)",
  mixed: "Textbook + research tools",
  emergency: "Emergency protocol",
  out_of_scope: "Out of scope",
  insufficient: "Insufficient evidence",
};

export const CITATION_SOURCE: Record<string, string> = {
  kb: "Textbook",
  pubmed: "PubMed",
  fda: "FDA label",
  medlineplus: "MedlinePlus",
  web: "Web",
};

export const TOOL_LABEL: Record<string, string> = {
  pubmed: "PubMed",
  drug_label: "openFDA",
  medlineplus: "MedlinePlus",
  web: "Web search",
};

export function formatScore(score: number): string {
  return score.toFixed(2);
}
