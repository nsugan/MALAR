export default function Card({ title, right, children, className = "" }) {
  return (
    <div className={`rounded-xl border border-slate-200 bg-white shadow-sm ${className}`}>
      {(title || right) && (
        <div className="flex items-center justify-between border-b border-slate-100 px-4 py-2.5">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</h2>
          {right}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}

export function Chip({ children, color = "slate" }) {
  const map = {
    slate: "bg-slate-100 text-slate-600 ring-1 ring-slate-200",
    green: "bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200",
    amber: "bg-amber-50 text-amber-700 ring-1 ring-amber-200",
    red: "bg-rose-50 text-rose-700 ring-1 ring-rose-200",
    indigo: "bg-indigo-50 text-indigo-700 ring-1 ring-indigo-200",
  };
  return <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${map[color]}`}>{children}</span>;
}

export function Button({ children, onClick, variant = "primary", disabled, className = "" }) {
  const v = {
    primary: "bg-indigo-600 hover:bg-indigo-500 text-white shadow-sm",
    ghost: "bg-white hover:bg-slate-50 text-slate-700 ring-1 ring-slate-300",
    green: "bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm",
    amber: "bg-amber-500 hover:bg-amber-400 text-white shadow-sm",
    red: "bg-rose-600 hover:bg-rose-500 text-white shadow-sm",
  }[variant];
  return (
    <button onClick={onClick} disabled={disabled}
      className={`rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-40 ${v} ${className}`}>
      {children}
    </button>
  );
}
