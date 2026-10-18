import { useEffect } from "react";

import { ChatView } from "./components/ChatView";
import { Composer } from "./components/Composer";
import { ErrorBanner } from "./components/ErrorBanner";
import { Inspector } from "./components/Inspector";
import { Sidebar } from "./components/Sidebar";
import { TopBar } from "./components/TopBar";
import { useTheme } from "./lib/theme";
import { useHealth } from "./lib/useHealth";
import { selectedResult, useChatStore } from "./store/chatStore";

export default function App() {
  const state = useChatStore();
  const { status, health } = useHealth();
  const [theme, toggleTheme] = useTheme();
  const result = selectedResult(state);
  const { refreshConversations } = state;

  useEffect(() => {
    void refreshConversations();
  }, [refreshConversations]);

  return (
    <div className="app">
      <TopBar
        status={status}
        theme={theme}
        onToggleTheme={toggleTheme}
        onNewConsultation={state.newConversation}
        busy={state.thinking}
      />

      <div className="workspace">
        <Sidebar
          conversations={state.conversations}
          activeId={state.threadId}
          disabled={state.thinking}
          onOpen={state.openConversation}
          onDelete={state.removeConversation}
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
            onSelect={state.select}
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
        />
      </div>
    </div>
  );
}
