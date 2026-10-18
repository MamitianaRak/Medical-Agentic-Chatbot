import { Square, Volume2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { synthesize } from "../api/client";
import { errorMessage } from "../store/chatStore";

interface Props {
  text: string;
  language: string;
  onError: (message: string) => void;
}

type State = "idle" | "loading" | "playing";

export function ListenButton({ text, language, onError }: Props) {
  const [state, setState] = useState<State>("idle");
  const audio = useRef<HTMLAudioElement | null>(null);

  useEffect(() => () => audio.current?.pause(), []);

  const toggle = async () => {
    if (state === "playing") {
      audio.current?.pause();
      setState("idle");
      return;
    }
    setState("loading");
    try {
      const url = URL.createObjectURL(await synthesize(text, language));
      const player = new Audio(url);
      player.onended = () => {
        URL.revokeObjectURL(url);
        setState("idle");
      };
      audio.current = player;
      await player.play();
      setState("playing");
    } catch (error) {
      setState("idle");
      onError(errorMessage(error));
    }
  };

  return (
    <button
      type="button"
      className="btn btn-ghost btn-sm"
      onClick={toggle}
      disabled={state === "loading"}
      aria-label={state === "playing" ? "Stop reading" : "Listen to a short summary"}
    >
      {state === "playing" ? <Square size={15} aria-hidden /> : <Volume2 size={16} aria-hidden />}
      {state === "loading" ? "Preparing audio…" : state === "playing" ? "Stop" : "Listen"}
    </button>
  );
}
