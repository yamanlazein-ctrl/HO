import { useNavigate } from "react-router-dom";
import { Camera, Video, MapPin, MonitorPlay } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getCameras } from "../lib/api";
import { cn } from "../lib/utils";

export function Cameras() {
  const { data: cameras, loading } = useFetch(getCameras, []);
  const navigate = useNavigate();

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div>
        <h1 className="text-xl font-bold text-[var(--text-primary)]">Cameras</h1>
        <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
          Camera sources & connection status. The pipeline is source-agnostic — iPhone/Camo today, RTSP IP camera tomorrow.
        </p>
      </div>

      {loading && <div className="text-[13px] text-[var(--text-muted)]">Loading cameras…</div>}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {cameras?.map((cam) => {
          const online = cam.status === "active";
          return (
            <button
              key={cam.id}
              onClick={() => navigate("/live")}
              className="panel group p-4 text-left transition-colors hover:border-[var(--accent)]/40 hover:bg-[var(--bg-hover)]"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className={cn("flex h-11 w-11 items-center justify-center rounded-lg", online ? "bg-[var(--accent)]/10" : "bg-[var(--bg-elevated)]")}>
                    <Camera className={cn("h-5 w-5", online ? "text-[var(--accent)]" : "text-[var(--text-muted)]")} />
                  </div>
                  <div>
                    <div className="text-[14px] font-bold text-[var(--text-primary)]">{cam.name}</div>
                    <div className="mono text-[11px] text-[var(--text-muted)]">CAM-{cam.id.toString().padStart(2, "0")}</div>
                  </div>
                </div>
                <span className={cn("flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase", online ? "bg-[var(--accent)]/15 text-[var(--accent)]" : "bg-[var(--bg-elevated)] text-[var(--text-muted)]")}>
                  <span className={cn("h-1.5 w-1.5 rounded-full", online ? "bg-[var(--accent)] animate-live-pulse" : "bg-[var(--text-muted)]")} />
                  {online ? "Online" : "Offline"}
                </span>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 border-t border-[var(--border-subtle)] pt-3 text-[11px]">
                <Spec icon={MapPin} label="Location" value={cam.location || "—"} />
                <Spec icon={Video} label="Source" value="iPhone / Camo" />
                <Spec icon={MonitorPlay} label="Resolution" value="1920×1080" />
                <Spec icon={MonitorPlay} label="FPS" value="30" />
              </div>

              <div className="mt-3 text-[11px] font-semibold text-[var(--accent)] opacity-0 transition-opacity group-hover:opacity-100">
                Open live monitoring →
              </div>
            </button>
          );
        })}

        {cameras?.length === 0 && !loading && (
          <div className="panel col-span-full flex flex-col items-center justify-center py-16">
            <Camera className="h-10 w-10 text-[var(--text-muted)]" />
            <p className="mt-3 text-[13px] text-[var(--text-secondary)]">No cameras registered</p>
            <p className="mt-1 text-[11px] text-[var(--text-muted)]">Add a camera via the API: POST /api/cameras</p>
          </div>
        )}
      </div>
    </div>
  );
}

function Spec({ icon: Icon, label, value }: { icon: typeof MapPin; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2">
      <Icon className="h-3.5 w-3.5 text-[var(--text-muted)]" />
      <div>
        <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
        <div className="mono text-[var(--text-secondary)]">{value}</div>
      </div>
    </div>
  );
}
