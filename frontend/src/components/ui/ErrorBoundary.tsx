/**
 * Top-level error boundary so a render-time crash in any view (especially
 * the canvas-heavy repo map) doesn't take the whole app down. Shows a
 * minimal recovery card with the message + a reload button.
 */
import { Component, ErrorInfo, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    // Surface to the dev console; intentionally not sent anywhere.
    // eslint-disable-next-line no-console
    console.error("[ErrorBoundary]", error, info.componentStack);
  }

  private handleReload = (): void => {
    this.setState({ error: null });
    if (typeof window !== "undefined") {
      window.location.reload();
    }
  };

  private handleDismiss = (): void => {
    this.setState({ error: null });
  };

  render(): ReactNode {
    const { error } = this.state;
    if (!error) {
      return this.props.children;
    }
    return (
      <div
        role="alert"
        style={{
          position: "fixed",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "rgba(10, 14, 26, 0.92)",
          zIndex: 9999,
          fontFamily: '"JetBrains Mono", monospace',
          color: "#e2e8f0",
          padding: 24,
        }}
      >
        <div
          style={{
            maxWidth: 540,
            width: "100%",
            background: "rgba(15, 23, 42, 0.95)",
            border: "1px solid rgba(239, 68, 68, 0.45)",
            borderRadius: 12,
            padding: "24px 28px",
            boxShadow: "0 20px 60px rgba(0,0,0,0.5)",
          }}
        >
          <div style={{ color: "#f87171", fontSize: 12, letterSpacing: 2 }}>
            UI ERROR
          </div>
          <h2
            style={{
              margin: "8px 0 16px",
              fontFamily: '"Orbitron", sans-serif',
              fontSize: 20,
              letterSpacing: 1,
            }}
          >
            Something went wrong rendering the UI.
          </h2>
          <pre
            style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              fontSize: 12,
              background: "rgba(0,0,0,0.35)",
              padding: 12,
              borderRadius: 6,
              maxHeight: 220,
              overflowY: "auto",
              marginBottom: 18,
              color: "#fca5a5",
            }}
          >
            {error.message}
            {error.stack ? `\n\n${error.stack}` : ""}
          </pre>
          <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
            <button
              type="button"
              onClick={this.handleDismiss}
              style={btnStyle(false)}
            >
              Dismiss
            </button>
            <button
              type="button"
              onClick={this.handleReload}
              style={btnStyle(true)}
            >
              Reload
            </button>
          </div>
        </div>
      </div>
    );
  }
}

function btnStyle(primary: boolean): React.CSSProperties {
  return {
    background: primary ? "rgba(0, 212, 255, 0.15)" : "transparent",
    border: `1px solid ${
      primary ? "rgba(0, 212, 255, 0.6)" : "rgba(255,255,255,0.18)"
    }`,
    color: primary ? "#00d4ff" : "#cbd5e1",
    padding: "6px 16px",
    borderRadius: 6,
    cursor: "pointer",
    fontSize: 12,
    letterSpacing: 1,
    fontFamily: '"JetBrains Mono", monospace',
  };
}
