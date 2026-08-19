import { useEffect, useState } from "react";
import { Activity, Cpu, Video, Gauge, Menu } from "lucide-react";
import { cn } from "../lib/utils";
import { getStatus } from "../lib/api";
import type { SystemStatus } from "../types";

function StatusDot({ ok, pulse }: { ok: boolean; pulse?: boolean }) {
  return (
    <span
      className={cn(
        "inline-block h-2 w-2 rounded-full",
        ok ? "bg-[var(--accent)]" : "bg-[var(--danger)]",
        pulse && ok && "animate-live-pulse",
      )}
    />
  );
}

function Metric({ icon: Icon, label, value, ok }: {
  icon: typeof Activity;
  label: string;
  value: string;
  ok: boolean;
}) {
  return (
    <div className="flex items-center gap-2 px-3">
      <Icon className={cn("h-4 w-4", ok ? "text-[var(--accent)]" : "text-[var(--text-muted)]")} />
      <div className="leading-tight">
        <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
        <div className={cn("mono text-[12px] font-semibold", ok ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]")}>
          {value}
        </div>
      </div>
    </div>
  );
}

export function Header({ onMenuClick }: { onMenuClick: () => void }) {
  const [status, setStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    let alive = true;
    const poll = () =>
      getStatus()
        .then((s) => alive && setStatus(s))
        .catch(() => alive && setStatus(null));
    poll();
    const t = setInterval(poll, 3000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  const aiOnline = status?.ai_engine.status === "online";
  const camOnline = status?.camera.status === "online";
  const fps = status?.processing.fps;
  const latency = status?.processing.latency_ms;

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center gap-2 border-b border-[var(--border-subtle)] bg-[var(--bg-panel)]/95 px-4 backdrop-blur">
      <button
        onClick={onMenuClick}
        className="mr-1 inline-flex h-9 w-9 items-center justify-center rounded-lg text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] lg:hidden"
        aria-label="Toggle navigation"
      >
        <Menu className="h-5 w-5" />
      </button>

      <div className="hidden items-center gap-2.5 sm:flex">
        <StatusDot ok={status?.system_online ?? false} pulse />
        <span className="text-[12px] font-semibold uppercase tracking-[0.14em] text-[var(--text-secondary)]">
          {status?.system_online ? "System Operational" : "System Offline"}
        </span>
      </div>

      <div className="ml-auto flex items-center gap-1 sm:gap-2">
        <div className="hidden items-center rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] py-1 md:flex">
          <Metric icon={Cpu} label="AI Engine" value={aiOnline ? "ONLINE" : "OFFLINE"} ok={aiOnline} />
          <div className="h-7 w-px bg-[var(--border-subtle)]" />
          <Metric icon={Video} label="Camera" value={camOnline ? "ONLINE" : "OFFLINE"} ok={camOnline} />
          <div className="h-7 w-px bg-[var(--border-subtle)]" />
          <Metric
            icon={Gauge}
            label="FPS"
            value={fps != null ? fps.toFixed(1) : "—"}
            ok={fps != null}
          />
          <div className="h-7 w-px bg-[var(--border-subtle)]" />
          <Metric
            icon={Activity}
            label="Latency"
            value={latency != null ? `${latency.toFixed(0)}ms` : "—"}
            ok={latency != null}
          />
        </div>
      </div>
    </header>
  );
}
