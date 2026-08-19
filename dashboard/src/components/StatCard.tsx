import type { LucideIcon } from "lucide-react";
import { cn } from "../lib/utils";

interface StatCardProps {
  label: string;
  value: string | number;
  sub?: string;
  icon: LucideIcon;
  accent?: "accent" | "danger" | "warning" | "info";
  loading?: boolean;
}

const ACCENT_MAP = {
  accent: { ring: "text-[var(--accent)]", bg: "bg-[var(--accent)]/10" },
  danger: { ring: "text-[var(--danger)]", bg: "bg-[var(--danger)]/10" },
  warning: { ring: "text-[var(--warning)]", bg: "bg-[var(--warning)]/10" },
  info: { ring: "text-[var(--info)]", bg: "bg-[var(--info)]/10" },
};

export function StatCard({ label, value, sub, icon: Icon, accent = "accent", loading }: StatCardProps) {
  const a = ACCENT_MAP[accent];
  return (
    <div className="panel flex items-center gap-4 p-4">
      <div className={cn("flex h-11 w-11 shrink-0 items-center justify-center rounded-lg", a.bg)}>
        <Icon className={cn("h-5 w-5", a.ring)} />
      </div>
      <div className="min-w-0">
        <div className="truncate text-[11px] font-medium uppercase tracking-wider text-[var(--text-muted)]">
          {label}
        </div>
        {loading ? (
          <div className="mt-1 h-6 w-16 animate-pulse rounded bg-[var(--bg-hover)]" />
        ) : (
          <div className="mono text-2xl font-bold leading-tight text-[var(--text-primary)]">
            {value}
          </div>
        )}
        {sub && <div className="mt-0.5 truncate text-[11px] text-[var(--text-secondary)]">{sub}</div>}
      </div>
    </div>
  );
}
