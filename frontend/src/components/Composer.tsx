import { ImagePlus, Mic, Send, Square, X } from "lucide-react";
import { useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";

import { transcribe } from "../api/client";
import { readAsDataUrl, startRecording } from "../lib/audio";
import { errorMessage } from "../store/chatStore";

const IMAGE_TYPES = ["image/png", "image/jpeg", "image/webp"];
const MAX_IMAGE_BYTES = 5_000_000;

interface Props {
  busy: boolean;
  onSend: (text: string, image: string | null) => void;
  onError: (message: string) => void;
}

type Recording = { stop: () => Promise<Blob> } | null;

export function Composer({ busy, onSend, onError }: Props) {
  const [text, setText] = useState("");
  const [image, setImage] = useState<string | null>(null);
  const [recording, setRecording] = useState<Recording>(null);
  const [transcribing, setTranscribing] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (busy || !text.trim()) return;
    onSend(text, image);
    setText("");
    setImage(null);
  };

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const pickImage = async (file: File | undefined) => {
    if (!file) return;
    if (!IMAGE_TYPES.includes(file.type) || file.size > MAX_IMAGE_BYTES) {
      onError("Use a PNG, JPEG or WebP image under 5 MB.");
      return;
    }
    setImage(await readAsDataUrl(file));
  };

  const toggleRecording = async () => {
    if (recording) {
      setRecording(null);
      setTranscribing(true);
      try {
        const transcript = await transcribe(await recording.stop());
        setText((current) => (current ? `${current} ${transcript}` : transcript));
      } catch (error) {
        onError(errorMessage(error));
      } finally {
        setTranscribing(false);
      }
      return;
    }
    try {
      setRecording(await startRecording());
    } catch {
      onError("Microphone unavailable. Allow microphone access in your browser.");
    }
  };

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="question" className="composer-label">
        Describe your symptoms or question
      </label>

      {image ? (
        <div className="attachment">
          <img src={image} alt="Selected for analysis" />
          <span className="small muted">The image is described objectively, never diagnosed.</span>
          <button
            type="button"
            className="icon-btn"
            onClick={() => setImage(null)}
            aria-label="Remove image"
          >
            <X size={16} />
          </button>
        </div>
      ) : null}

      <div className="composer-box">
        <textarea
          id="question"
          rows={2}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            transcribing ? "Transcribing your question…" : "e.g. I have had a dry cough for five days…"
          }
          maxLength={4000}
          disabled={transcribing}
        />
        <input
          ref={fileInput}
          type="file"
          accept={IMAGE_TYPES.join(",")}
          hidden
          onChange={(event) => {
            void pickImage(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
        <button
          type="button"
          className="tool-btn"
          onClick={() => fileInput.current?.click()}
          disabled={busy}
          aria-label="Attach an image"
        >
          <ImagePlus size={18} />
        </button>
        <button
          type="button"
          className={`tool-btn${recording ? " recording" : ""}`}
          onClick={toggleRecording}
          disabled={busy || transcribing}
          aria-label={recording ? "Stop recording" : "Record a voice question"}
        >
          {recording ? <Square size={16} /> : <Mic size={18} />}
        </button>
        <button type="submit" className="btn btn-primary" disabled={busy || !text.trim()}>
          <Send size={16} aria-hidden />
          Ask
        </button>
      </div>
      <span className="small muted">
        Educational information only. Not a diagnosis. In an emergency, call your local emergency
        number.
      </span>
    </form>
  );
}
