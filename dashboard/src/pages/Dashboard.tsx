import { Link } from "react-router-dom";
import {
  AlertTriangle,
  Camera,
  CheckCircle2,
  Gauge,
  TrendingUp,
  Video,
  Cpu,
} from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getCameras, getEvents, getStatistics, getStatus } from "../lib/api";
import { StatCard } from "../components/StatCard";
import { LiveCamera } from "../components/LiveCamera";
import { Badge } from "../components/Badge";
import { cn, formatTime, formatConfidence } from "../lib/utils";
import type { Event } from "../types";

export function Dashboard() {
  const { data: cameras } = useFetch(getCameras, []);
  const { data: eventsList } = useFetch(() => getEvents(8, 0), [], 3000);
  const { data: stats } = useFetch(getStatistics, [], 5000);
  const { data: status } = useFetch(getStatus, [], 2000);

  const events: Event[] = eventsList?.items ?? [];
  const primaryCam = cameras?.[0];

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div>
        <h1 className="text-xl font-bold text-[var(--text-primary)]">Dashboard</h1>
        <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
          Real-time AI littering detection & evidence collection
        </p>
      </div>

      {/* Status area */}
      <div className="panel grid grid-cols-2 gap-px overflow-hidden bg-[var(--border-subtle)] sm:grid-cols-4">
        {[
          { label: "System", value: status?.system_online ? "Online" : "Offline", ok: status?.system_online, icon: CheckCircle2 },
          { label: "AI Engine", value: status?.ai_engine.status ?? "—", ok: status?.ai_engine.status === "online", icon: Cpu },
          { label: "Camera", value: status?.camera.status ?? "—", ok: status?.camera.status === "online", icon: Video },
          { label: "Processing", value: status?.processing.fps != null ? `${status.processing.fps.toFixed(0)} FPS` : "Idle", ok: status?.processing.fps != null, icon: Gauge },
        ].map((s) => (
          <div key={s.label} className="flex items-center gap-3 bg-[var(--bg-panel)] px-4 py-3">
            <s.icon className={cn("h-5 w-5", s.ok ? "text-[var(--accent)]" : "text-[var(--text-muted)]")} />
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">{s.label}</div>
              <div className={cn("mono text-[13px] font-bold capitalize", s.ok ? "text-[var(--text-primary)]" : "text-[var(--text-muted)]")}>
                {s.value}
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Main area: live camera + side info */}
      <div className="grid gap-6 xl:grid-cols-[1fr_360px]">
        <div className="space-y-4">
          {primaryCam ? (
            <LiveCamera
              cameraId={primaryCam.id}
              cameraName={primaryCam.name}
              aiState={status?.live_state?.ai_state}
              entities={status?.live_state?.entities || []}
            />
          ) : (
            <div className="panel flex aspect-video flex-col items-center justify-center">
              <Camera className="h-10 w-10 text-[var(--text-muted)]" />
              <p className="mt-3 text-[13px] text-[var(--text-secondary)]">No cameras registered</p>
              <Link to="/cameras" className="mt-2 text-[12px] font-semibold text-[var(--accent)] hover:underline">
                Add a camera →
              </Link>
            </div>
          )}

          {/* Stats row */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Violations Today" value={stats?.events_today ?? 0} icon={AlertTriangle} accent="danger" loading={!stats} />
            <StatCard label="Active Cameras" value={cameras?.length ?? 0} icon={Camera} loading={!cameras} />
            <StatCard label="Total Events" value={stats?.total_events ?? 0} icon={TrendingUp} accent="info" loading={!stats} />
            <StatCard
              label="Avg Confidence"
              value={stats?.avg_confidence ? formatConfidence(stats.avg_confidence) : "—"}
              icon={Gauge}
              accent="warning"
              loading={!stats}
            />
          </div>
        </div>

        {/* Side: recent violations */}
        <div className="panel flex flex-col">
          <div className="flex items-center justify-between border-b border-[var(--border-subtle)] px-4 py-3">
            <h2 className="text-[13px] font-semibold text-[var(--text-primary)]">Recent Violations</h2>
            <Link to="/violations" className="text-[11px] font-semibold text-[var(--accent)] hover:underline">
              View all
            </Link>
          </div>
          <div className="divide-y divide-[var(--border-subtle)]">
            {events.length === 0 && (
              <div className="px-4 py-10 text-center text-[12px] text-[var(--text-muted)]">
                No violations detected yet
              </div>
            )}
            {events.map((ev) => (
              <Link
                key={ev.id}
                to={`/violations/${ev.id}`}
                className="flex items-center gap-3 px-4 py-3 transition-colors hover:bg-[var(--bg-hover)]"
              >
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--danger)]/12">
                  <AlertTriangle className="h-4 w-4 text-[var(--danger)]" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="mono text-[12px] font-bold text-[var(--text-primary)]">#{ev.id}</span>
                    <span className="truncate text-[12px] text-[var(--text-secondary)]">{ev.object_type}</span>
                  </div>
                  <div className="mt-0.5 text-[11px] text-[var(--text-muted)]">
                    {formatTime(ev.timestamp)} · {formatConfidence(ev.confidence)}
                  </div>
                </div>
                <Badge status={ev.status} />
              </Link>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
