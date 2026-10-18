import { useEffect, useState } from "react";

import { getHealth } from "../api/client";
import type { Health } from "../api/schemas";

export type ApiStatus = "checking" | "online" | "offline";

export function useHealth(intervalMs = 30_000): { status: ApiStatus; health: Health | null } {
  const [state, setState] = useState<{ status: ApiStatus; health: Health | null }>({
    status: "checking",
    health: null,
  });

  useEffect(() => {
    const controller = new AbortController();
    const probe = async () => {
      const health = await getHealth(controller.signal);
      if (!controller.signal.aborted) {
        setState({ status: health ? "online" : "offline", health });
      }
    };

    probe();
    const timer = setInterval(probe, intervalMs);
    return () => {
      controller.abort();
      clearInterval(timer);
    };
  }, [intervalMs]);

  return state;
}
