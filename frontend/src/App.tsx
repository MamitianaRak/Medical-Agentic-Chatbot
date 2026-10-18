import { useEffect, useState } from "react";

import { ChatView } from "./components/ChatView";
import { Composer } from "./components/Composer";
import { ErrorBanner } from "./components/ErrorBanner";
import { Inspector } from "./components/Inspector";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { useTheme } from "./lib/theme";
import { useHealth } from "./lib/useHealth";
import { selectedResult, useChatStore } from "./store/chatStore";

/** Side panels become drawers on small screens; on wide screens they are always visible. */
type Drawer = "none" | "conversations" | "inspector";

export default function App() {
  const state = useChatStore();
  const { status, health } = useHealth();
  const [theme, toggleTheme] = useTheme();
  const [drawer, setDrawer] = useState<Drawer>("none");
  const result = selectedResult(state);
  const { refreshConversations } = state;
  const closeDrawer = () => setDrawer("none");

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  useEffect(() => {
    if (drawer === "none") return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setDrawer("none");
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [drawer]);

  return (
    <div className="app" data-drawer={drawer}>
      <TopBar
        status={status}
        theme={theme}
        onToggleTheme={toggleTheme}
        onNewConsultation={() => {
          state.newConversation();
          closeDrawer();
        }}
        onOpenConversations={() => setDrawer("conversations")}
        onOpenInspector={() => setDrawer("inspector")}
        busy={state.thinking}
      />

      <div className="workspace">
        <button
          type="button"
          className="backdrop"
          onClick={closeDrawer}
          aria-label="Close panel"
          tabIndex={-1}
        />

        <Sidebar
          conversations={state.conversations}
          activeId={state.threadId}
          disabled={state.thinking}
          onOpen={(threadId) => {
            closeDrawer();
            void state.openConversation(threadId);
          }}
          onDelete={state.removeConversation}
          onClose={closeDrawer}
        />

        <main className="chat">
          {state.error ? (
            <ErrorBanner message={state.error} onDismiss={() => state.setError(null)} />
          ) : null}
          <ChatView
            messages={state.messages}
            thinking={state.thinking}
            liveSteps={state.liveSteps}
            selectedId={state.selectedId}
            onSelect={(messageId) => {
              state.select(messageId);
              setDrawer("inspector");
            }}
            onExample={(question) => void state.send(question)}
            onError={state.setError}
          />
          <Composer
            busy={state.thinking}
            onSend={(text, image) => void state.send(text, image)}
            onError={state.setError}
          />
        </main>

        <Inspector
          result={result}
          liveSteps={state.liveSteps}
          thinking={state.thinking}
          models={health?.models ?? {}}
          onClose={closeDrawer}
        />
      </div>
    </div>
  );
}
