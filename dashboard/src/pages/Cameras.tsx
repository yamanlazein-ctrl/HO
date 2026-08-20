import { useNavigate } from "react-router-dom";
import { Camera, Video, MapPin, MonitorPlay, Smartphone, CheckCircle2, AlertCircle } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getCameras, getStatus } from "../lib/api";
import { cn } from "../lib/utils";

export function Cameras() {
  const { data: cameras, loading } = useFetch(getCameras, [], 3000);
  const { data: status } = useFetch(getStatus, [], 2000);
  const navigate = useNavigate();

  const camOnline = status?.camera.status === "online";
  const sourceName = status?.camera.source ?? "None";
  const camFps = status?.camera.fps;

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-[var(--text-primary)]">Cameras & Device Setup</h1>
          <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
            Manage physical camera feeds, phone integration (iPhone / Android / Camo), and RTSP devices.
          </p>
        </div>
      </div>

      {/* Interactive Phone / Camera Onboarding Steps */}
      <div className="panel p-5 border-[var(--border-default)] bg-[var(--bg-panel)]">
        <div className="flex items-center gap-2 mb-4">
          <Smartphone className="h-5 w-5 text-[var(--accent)]" />
          <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">
            Interactive Camera Setup Wizard
          </h2>
        </div>

        <div className="grid gap-4 md:grid-cols-4">
          {/* Step 1 */}
          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--accent)]">Step 1</span>
                <CheckCircle2 className="h-4 w-4 text-[var(--accent)]" />
              </div>
              <h3 className="text-xs font-semibold text-[var(--text-primary)] mb-1">Install Camo / Iriun</h3>
              <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                Open App Store or Play Store on your phone and install <strong>Camo</strong> or <strong>Iriun Webcam</strong>.
              </p>
            </div>
            <div className="mt-3 text-[10px] text-[var(--text-muted)]">iPhone & Android supported</div>
          </div>

          {/* Step 2 */}
          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--accent)]">Step 2</span>
                <span className={cn("h-2 w-2 rounded-full", camOnline ? "bg-[var(--accent)]" : "bg-[var(--warning)] animate-live-pulse")} />
              </div>
              <h3 className="text-xs font-semibold text-[var(--text-primary)] mb-1">Connect USB Cable</h3>
              <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                Plug phone into your PC via USB and start Camo Studio. Keep phone unlocked.
              </p>
            </div>
            <div className="mt-3 text-[10px] text-[var(--text-muted)]">DirectShow / UVC Stream</div>
          </div>

          {/* Step 3 */}
          <div className={cn(
            "rounded-lg border p-4 flex flex-col justify-between transition-all",
            camOnline 
              ? "border-[var(--accent)] bg-[var(--accent)]/10" 
              : "border-[var(--border-subtle)] bg-[var(--bg-elevated)]"
          )}>
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--accent)]">Step 3</span>
                {camOnline ? (
                  <CheckCircle2 className="h-4 w-4 text-[var(--accent)]" />
                ) : (
                  <AlertCircle className="h-4 w-4 text-[var(--warning)]" />
                )}
              </div>
              <h3 className="text-xs font-semibold text-[var(--text-primary)] mb-1">Live Feed Detection</h3>
              <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                {camOnline ? (
                  <span className="text-[var(--accent)] font-medium">Device connected & streaming live!</span>
                ) : (
                  <span>Waiting for camera signal from your device...</span>
                )}
              </p>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <span className="mono text-[10px] text-[var(--text-muted)]">
                {camOnline ? `${camFps?.toFixed(0) || 30} FPS` : "OFFLINE"}
              </span>
              <span className={cn(
                "text-[10px] font-bold uppercase px-2 py-0.5 rounded",
                camOnline ? "bg-[var(--accent)] text-black" : "bg-black/30 text-[var(--text-muted)]"
              )}>
                {camOnline ? "CONNECTED" : "WAITING"}
              </span>
            </div>
          </div>

          {/* Step 4 */}
          <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-4 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--accent)]">Step 4</span>
                <MonitorPlay className="h-4 w-4 text-[var(--accent)]" />
              </div>
              <h3 className="text-xs font-semibold text-[var(--text-primary)] mb-1">Start Monitoring</h3>
              <p className="text-[11px] text-[var(--text-secondary)] leading-relaxed">
                Navigate to Live Monitoring to inspect live YOLO bounding boxes and temporal tracking.
              </p>
            </div>
            <button
              onClick={() => navigate("/live")}
              className="mt-3 w-full rounded bg-[var(--accent)] py-1.5 text-center text-xs font-bold text-black hover:bg-[var(--accent-dim)] transition-colors"
            >
              Open Live View →
            </button>
          </div>
        </div>
      </div>

      {/* Camera Grid List */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)]">
            Registered Cameras ({cameras?.length || 0})
          </h2>
        </div>

        {loading && <div className="text-[13px] text-[var(--text-muted)]">Loading cameras…</div>}

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {cameras?.map((cam) => {
            const online = camOnline || cam.status === "active";
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
                  <Spec icon={MapPin} label="Location" value={cam.location || "Main Entrance"} />
                  <Spec icon={Video} label="Source" value={sourceName} />
                  <Spec icon={MonitorPlay} label="Resolution" value="1920×1080" />
                  <Spec icon={MonitorPlay} label="FPS" value={camFps ? camFps.toFixed(0) : "30"} />
                </div>

                <div className="mt-3 text-[11px] font-semibold text-[var(--accent)] opacity-0 transition-opacity group-hover:opacity-100">
                  Open live monitoring →
                </div>
              </button>
            );
          })}

          {(!cameras || cameras.length === 0) && !loading && (
            <div className="panel col-span-full flex flex-col items-center justify-center py-16">
              <Camera className="h-10 w-10 text-[var(--text-muted)]" />
              <p className="mt-3 text-[13px] text-[var(--text-secondary)]">No cameras registered</p>
              <p className="mt-1 text-[11px] text-[var(--text-muted)]">The pipeline will automatically register your default camera on start.</p>
            </div>
          )}
        </div>
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
