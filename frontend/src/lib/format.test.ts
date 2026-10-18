import { describe, expect, it } from "vitest";

import { withCitationChips } from "./format";

describe("withCitationChips", () => {
  it("turns single and grouped PDF citations into inline code chips", () => {
    expect(withCitationChips("Smoke [PDF p.394]. Pollen [PDF p.250, PDF p.394].")).toBe(
      "Smoke `PDF p.394`. Pollen `PDF p.250` `PDF p.394`.",
    );
  });

  it("handles research tool labels, alone or grouped", () => {
    expect(
      withCitationChips("A [PMID:123; FDA: ibuprofen]. B [MedlinePlus: Asthma] [Web: who.int]."),
    ).toBe("A `PMID:123` `FDA: ibuprofen`. B `MedlinePlus: Asthma` `Web: who.int`.");
  });

  it("leaves URLs and other brackets untouched", () => {
    const text = "See [https://example.org] and [note].";
    expect(withCitationChips(text)).toBe(text);
  });
});
