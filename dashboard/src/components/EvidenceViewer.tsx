import { useState } from "react";
import { ImageOff, Film, Clock } from "lucide-react";
import type { Evidence } from "../types";
import { cn } from "../lib/utils";
import { evidenceFileUrl } from "../lib/api";

interface Props {
  evidence: Evidence[];
  /** show the before/event/after segment labels under the video */
  showSegments?: boolean;
}

export function EvidenceViewer({ evidence, showSegments }: Props) {
  const [selected, setSelected] = useState(0);

  if (evidence.length === 0) {
    return (
      <div className="panel flex flex-col items-center justify-center py-20">
        <ImageOff className="h-10 w-10 text-[var(--text-muted)]" />
        <p className="mt-3 text-[13px] text-[var(--text-secondary)]">No evidence uploaded</p>
      </div>
    );
  }

  const current = evidence[selected];

  return (
    <div className="space-y-4">
      <div className="panel overflow-hidden">
        {evidence.length > 1 && (
          <div className="flex border-b border-[var(--border-subtle)]">
            {evidence.map((ev, idx) => (
              <button
                key={ev.id}
                onClick={() => setSelected(idx)}
                className={cn(
                  "px-4 py-2.5 text-[12px] font-medium transition-colors",
                  idx === selected
                    ? "border-b-2 border-[var(--accent)] text-[var(--text-primary)]"
                    : "text-[var(--text-muted)] hover:text-[var(--text-secondary)]",
                )}
              >
                Evidence #{ev.id}
              </button>
            ))}
          </div>
        )}

        <div className="grid gap-4 p-4 md:grid-cols-2">
          {/* Snapshot */}
          <div>
            <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              <ImageOff className="h-3.5 w-3.5" /> Snapshot
            </div>
            <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)]">
              {current.image_path ? (
                <img
                  src={evidenceFileUrl(current.image_path)}
                  alt="Event snapshot"
                  className="h-full w-full object-contain"
                  onError={(e) => ((e.target as HTMLImageElement).style.opacity = "0.2")}
                />
              ) : (
                <div className="flex flex-col items-center text-[var(--text-muted)]">
                  <ImageOff className="h-8 w-8" />
                  <span className="mt-2 text-[11px]">No snapshot</span>
                </div>
              )}
            </div>
          </div>

          {/* Video */}
          <div>
            <div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">
              <Film className="h-3.5 w-3.5" /> Evidence Video
            </div>
            <div className="flex aspect-video items-center justify-center overflow-hidden rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)]">
              {current.video_path ? (
                <video
                  src={evidenceFileUrl(current.video_path)}
                  controls
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center text-[var(--text-muted)]">
                  <Film className="h-8 w-8" />
                  <span className="mt-2 text-[11px]">No video</span>
                </div>
              )}
            </div>
          </div>
        </div>

        {showSegments && (
          <div className="grid grid-cols-3 gap-px border-t border-[var(--border-subtle)] bg-[var(--border-subtle)]">
            {["Before Event", "Event", "After Event"].map((label, i) => (
              <div key={label} className="bg-[var(--bg-panel)] px-3 py-2 text-center">
                <div
                  className={cn(
                    "text-[10px] font-semibold uppercase tracking-wider",
                    i === 1 ? "text-[var(--danger)]" : "text-[var(--text-muted)]",
                  )}
                >
                  {label}
                </div>
                <div className="mono mt-0.5 text-[11px] text-[var(--text-secondary)]">
                  {current.duration_sec ? `${(current.duration_sec / 3).toFixed(1)}s` : "—"}
                </div>
              </div>
            ))}
          </div>
        )}

        <div className="flex items-center gap-6 border-t border-[var(--border-subtle)] px-4 py-3 text-[11px] text-[var(--text-muted)]">
          <div className="flex items-center gap-1.5">
            <Clock className="h-3.5 w-3.5" />
            {current.duration_sec != null ? `${current.duration_sec.toFixed(1)}s clip` : "Duration —"}
          </div>
          <div>
            Image: <span className="mono text-[var(--text-secondary)]">{current.image_path ?? "—"}</span>
          </div>
        </div>
      </div>
    </div>
  );
}
