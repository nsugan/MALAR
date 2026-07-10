import { Component } from "react";

// Per-tab error boundary. Without this, an uncaught render error in ONE tab unmounts the
// whole React tree and blanks the page until a full reload. With it, the error is caught,
// shown inline, and the header/nav stay live — so switching tabs (which remounts this
// boundary via its `key`) recovers immediately, no reload needed.
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // surface to the console for debugging; never rethrow
    // eslint-disable-next-line no-console
    console.error("[MALAR] tab render error:", error, info);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      const msg = String(this.state.error?.message || this.state.error);
      return (
        <div className="rounded-lg border border-rose-200 bg-rose-50 p-4 text-sm">
          <div className="font-semibold text-rose-700">This view hit an error.</div>
          <pre className="mt-2 max-h-40 overflow-auto whitespace-pre-wrap text-[11px] text-rose-600">{msg}</pre>
          <div className="mt-2 flex items-center gap-2">
            <button onClick={this.reset}
              className="rounded bg-rose-600 px-3 py-1 text-white hover:bg-rose-500">Retry</button>
            <span className="text-xs text-slate-500">…or pick another tab — the app stays loaded.</span>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
