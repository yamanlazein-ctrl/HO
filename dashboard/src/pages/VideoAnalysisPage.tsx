import React, { useRef, useState } from "react";
import {
  Upload,
  FileVideo,
  Layers,
  FileText,
  AlertCircle
} from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getAnalysisJobs, uploadVideoAnalysis } from "../lib/api";
import { cn, formatTime } from "../lib/utils";

export function VideoAnalysisPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [activeJobId, setActiveJobId] = useState<number | null>(null);

  // Poll analysis jobs list every 3s
  const { data: jobsData, loading } = useFetch(() => getAnalysisJobs(20, 0), [], 3000);
  const jobs = jobsData?.items ?? [];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
      setUploadError(null);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) return;
    setUploading(true);
    setUploadError(null);
    try {
      const job = await uploadVideoAnalysis(selectedFile);
      setActiveJobId(job.id);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  const selectedJob = jobs.find((j) => j.id === activeJobId) ?? jobs[0];

  let parsedReport: any = null;
  if (selectedJob?.report_json) {
    try {
      parsedReport = JSON.parse(selectedJob.report_json);
    } catch {}
  }

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-[var(--text-primary)]">Video File Analysis</h1>
          <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
            Upload and analyze recorded CCTV / benchmark videos through the full production AI pipeline.
          </p>
        </div>
      </div>

      {/* Upload Zone & Job Control */}
      <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
        {/* Upload & Active Execution Card */}
        <div className="panel p-5 space-y-5">
          <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)] flex items-center gap-2">
            <Upload className="h-4 w-4 text-[var(--accent)]" /> Upload Video For Full Analysis
          </h2>

          <div
            onClick={() => fileInputRef.current?.click()}
            className="border-2 border-dashed border-[var(--border-default)] hover:border-[var(--accent)] rounded-xl p-8 flex flex-col items-center justify-center cursor-pointer transition-colors bg-[var(--bg-elevated)]"
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp4,.avi,.mov,.mkv,.webm"
              className="hidden"
              onChange={handleFileChange}
            />
            <FileVideo className="h-10 w-10 text-[var(--accent)] mb-3" />
            <p className="text-sm font-semibold text-[var(--text-primary)]">
              {selectedFile ? selectedFile.name : "Click to browse or drop video file"}
            </p>
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Supports .mp4, .avi, .mov, .mkv (Max 200MB recommended)
            </p>
          </div>

          {selectedFile && (
            <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-base)] p-4 flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-[var(--text-primary)]">{selectedFile.name}</div>
                <div className="text-[11px] text-[var(--text-muted)]">
                  Size: {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </div>
              </div>
              <button
                disabled={uploading}
                onClick={handleStartAnalysis}
                className="rounded-lg bg-[var(--accent)] px-4 py-2 text-xs font-bold text-black hover:bg-[var(--accent-dim)] transition-colors disabled:opacity-50"
              >
                {uploading ? "Uploading & Starting..." : "Start AI Analysis →"}
              </button>
            </div>
          )}

          {uploadError && (
            <div className="rounded-lg bg-[var(--danger)]/15 border border-[var(--danger)]/30 p-3 text-xs text-[var(--danger)] flex items-center gap-2">
              <AlertCircle className="h-4 w-4 shrink-0" /> {uploadError}
            </div>
          )}

          {/* Active Job Real-Time Progress View */}
          {selectedJob && (
            <div className="border-t border-[var(--border-subtle)] pt-4 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--accent)]">
                    Active Job #{selectedJob.id}
                  </span>
                  <h3 className="text-sm font-bold text-[var(--text-primary)]">{selectedJob.original_filename}</h3>
                </div>
                <span
                  className={cn(
                    "rounded-full px-2.5 py-0.5 text-[10px] font-bold uppercase",
                    selectedJob.status === "completed"
                      ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                      : selectedJob.status === "processing"
                      ? "bg-[var(--warning)]/15 text-[var(--warning)] animate-pulse"
                      : selectedJob.status === "failed"
                      ? "bg-[var(--danger)]/15 text-[var(--danger)]"
                      : "bg-[var(--bg-elevated)] text-[var(--text-muted)]"
                  )}
                >
                  {selectedJob.status}
                </span>
              </div>

              {/* Real Progress Bar */}
              {selectedJob.total_frames && selectedJob.total_frames > 0 ? (
                <div>
                  <div className="flex items-center justify-between text-xs text-[var(--text-secondary)] mb-1">
                    <span>
                      Frames: {selectedJob.processed_frames} / {selectedJob.total_frames}
                    </span>
                    <span className="mono">
                      {Math.round((selectedJob.processed_frames / selectedJob.total_frames) * 100)}%
                    </span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-[var(--bg-base)] overflow-hidden">
                    <div
                      className="h-full bg-[var(--accent)] transition-all duration-300"
                      style={{
                        width: `${Math.min(100, Math.round((selectedJob.processed_frames / selectedJob.total_frames) * 100))}%`
                      }}
                    />
                  </div>
                </div>
              ) : null}

              {/* Job Metrics Row */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--text-primary)]">
                    {selectedJob.duration_sec ? `${selectedJob.duration_sec.toFixed(1)}s` : "—"}
                  </div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Video Length</div>
                </div>
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--text-primary)]">
                    {selectedJob.processing_fps ? `${selectedJob.processing_fps.toFixed(1)} FPS` : "—"}
                  </div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Processing Speed</div>
                </div>
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--text-primary)]">{selectedJob.persons_detected}</div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Persons Tracked</div>
                </div>
                <div className="rounded border border-[var(--border-subtle)] bg-[var(--bg-base)] p-2">
                  <div className="mono text-xs font-bold text-[var(--danger)]">{selectedJob.events_count}</div>
                  <div className="text-[9px] uppercase tracking-wider text-[var(--text-muted)]">Littering Events</div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Diagnostic Report Panel */}
        <div className="panel p-5 space-y-4">
          <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)] flex items-center gap-2">
            <FileText className="h-4 w-4 text-[var(--accent)]" /> Diagnostic Inspection
          </h2>

          {parsedReport ? (
            <div className="space-y-4 text-xs">
              <div className="rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-elevated)] p-3 space-y-2">
                <div className="font-bold text-[var(--text-primary)] mb-1">Pipeline Stages Verification:</div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">YOLO Person Detection:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.yolo_person === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.yolo_person}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">YOLO Object Detection:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.yolo_object === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.yolo_object}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">ByteTrack Tracking:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.tracking === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.tracking}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[var(--text-secondary)]">Person-Object Association:</span>
                  <span className={cn("font-bold", parsedReport.diagnosis.association === "PASS" ? "text-[var(--accent)]" : "text-[var(--danger)]")}>
                    {parsedReport.diagnosis.association}
                  </span>
                </div>
                <div className="flex justify-between border-t border-[var(--border-subtle)] pt-1">
                  <span className="text-[var(--text-secondary)]">Final Outcome:</span>
                  <span className={cn("font-bold", parsedReport.confirmed_events > 0 ? "text-[var(--danger)]" : "text-[var(--text-muted)]")}>
                    {parsedReport.diagnosis.littering_candidate}
                  </span>
                </div>
              </div>

              {/* Timeline progression */}
              {parsedReport.timeline && parsedReport.timeline.length > 0 && (
                <div>
                  <div className="font-bold text-[var(--text-primary)] mb-2">Behavior Timeline:</div>
                  <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                    {parsedReport.timeline.map((item: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between rounded bg-[var(--bg-base)] p-1.5 text-[11px]">
                        <span className="mono text-[var(--accent)]">{item.timestamp}s</span>
                        <span className="mono font-semibold">{item.state}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-[var(--text-muted)]">
              Select or run a video analysis job to see step-by-step diagnostic breakdown.
            </div>
          )}
        </div>
      </div>

      {/* Analysis Jobs History Table */}
      <div className="panel p-5 space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wider text-[var(--text-primary)] flex items-center gap-2">
          <Layers className="h-4 w-4 text-[var(--accent)]" /> Video Analysis History
        </h2>

        {jobs.length === 0 && !loading && (
          <div className="py-12 text-center text-xs text-[var(--text-muted)]">
            No video analysis jobs uploaded yet. Upload an MP4 above to start.
          </div>
        )}

        {jobs.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-[var(--border-subtle)] text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                <tr>
                  <th className="px-3 py-2">ID</th>
                  <th className="px-3 py-2">File</th>
                  <th className="px-3 py-2">Status</th>
                  <th className="px-3 py-2">Frames</th>
                  <th className="px-3 py-2">Speed</th>
                  <th className="px-3 py-2">Persons</th>
                  <th className="px-3 py-2">Events</th>
                  <th className="px-3 py-2">Date</th>
                  <th className="px-3 py-2">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)]">
                {jobs.map((j) => (
                  <tr key={j.id} className="hover:bg-[var(--bg-hover)] transition-colors">
                    <td className="px-3 py-2.5 mono font-bold text-[var(--accent)]">#{j.id}</td>
                    <td className="px-3 py-2.5 font-medium text-[var(--text-primary)] max-w-[200px] truncate">
                      {j.original_filename}
                    </td>
                    <td className="px-3 py-2.5">
                      <span
                        className={cn(
                          "rounded px-2 py-0.5 text-[10px] font-bold uppercase",
                          j.status === "completed"
                            ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                            : j.status === "processing"
                            ? "bg-[var(--warning)]/15 text-[var(--warning)]"
                            : j.status === "failed"
                            ? "bg-[var(--danger)]/15 text-[var(--danger)]"
                            : "bg-[var(--bg-elevated)] text-[var(--text-muted)]"
                        )}
                      >
                        {j.status}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 mono">{j.processed_frames} / {j.total_frames || "—"}</td>
                    <td className="px-3 py-2.5 mono">{j.processing_fps ? `${j.processing_fps} FPS` : "—"}</td>
                    <td className="px-3 py-2.5 mono">{j.persons_detected}</td>
                    <td className="px-3 py-2.5 mono font-bold text-[var(--danger)]">{j.events_count}</td>
                    <td className="px-3 py-2.5 text-[var(--text-muted)]">{formatTime(j.created_at)}</td>
                    <td className="px-3 py-2.5">
                      <button
                        onClick={() => setActiveJobId(j.id)}
                        className="rounded bg-[var(--bg-elevated)] px-2.5 py-1 text-[11px] font-semibold text-[var(--text-primary)] hover:bg-[var(--border-default)] transition-colors"
                      >
                        Inspect
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
