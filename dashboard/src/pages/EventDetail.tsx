import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Camera, Clock, User, Package, Gauge, CheckCircle2, ShieldAlert } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getCamera, getEvent, getEvidence } from "../lib/api";
import { EvidenceViewer } from "../components/EvidenceViewer";
import { Badge } from "../components/Badge";
import { EventTimeline } from "../components/EventTimeline";
import { cn, formatDate, formatConfidence } from "../lib/utils";

export function EventDetail() {
  const { id } = useParams<{ id: string }>();
  const eventId = Number(id);

  const { data: event, loading, error } = useFetch(() => getEvent(eventId), [eventId]);
  const { data: camera } = useFetch(() => getCamera(event!.camera_id).catch(() => null), [event?.camera_id]);
  const { data: evidence } = useFetch(() => getEvidence(eventId), [eventId]);

  if (loading) {
    return <div className="p-7 text-[13px] text-[var(--text-muted)]">Loading event…</div>;
  }
  if (error || !event) {
    return (
      <div className="p-7">
        <Link to="/violations" className="text-[13px] font-semibold text-[var(--accent)] hover:underline">← Back to violations</Link>
        <p className="mt-4 text-[13px] text-[var(--danger)]">{error ?? "Event not found"}</p>
      </div>
    );
  }

  const isConfirmed = event.status === "confirmed";

  // Derive AI reasoning checklist from the event state.
  // The FSM reaches LITTERING_CONFIRMED only after all these hold.
  const reasoning = isConfirmed
    ? {
        HOLDING: true,
        RELEASE: true,
        OBJECT_ON_GROUND: true,
        PERSON_AWAY: true,
        LITTERING_CONFIRMED: true,
      }
    : undefined;

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <Link to="/violations" className="inline-flex items-center gap-1.5 text-[13px] font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)]">
        <ArrowLeft className="h-4 w-4" /> Back to violations
      </Link>

      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="mono text-2xl font-bold text-[var(--text-primary)]">Violation #{event.id}</h1>
            <Badge status={event.status} />
          </div>
          <p className="mt-1 text-[13px] text-[var(--text-secondary)]">
            {formatDate(event.timestamp)}
          </p>
        </div>
        {isConfirmed && (
          <div className="flex items-center gap-2 rounded-lg bg-[var(--danger)]/15 px-4 py-2.5">
            <ShieldAlert className="h-5 w-5 text-[var(--danger)]" />
            <span className="text-[13px] font-bold text-[var(--danger)]">LITTERING CONFIRMED</span>
          </div>
        )}
      </div>

      {/* Main grid: evidence (left) + info (right) */}
      <div className="grid gap-6 xl:grid-cols-[1fr_380px]">
        <div>
          <EvidenceViewer evidence={evidence ?? []} showSegments />
        </div>

        {/* Right column: metadata + AI reasoning */}
        <div className="space-y-4">
          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">Event Information</h2>
            <div className="space-y-2.5">
              <Field icon={Camera} label="Camera" value={camera?.name ?? `CAM-${event.camera_id}`} />
              <Field icon={Clock} label="Time" value={formatDate(event.timestamp)} />
              <Field icon={User} label="Person" value={event.person_track_id ? `Track #${event.person_track_id}` : "—"} mono />
              <Field icon={Package} label="Object" value={event.object_type} />
              <Field icon={Gauge} label="Confidence" value={formatConfidence(event.confidence)} mono />
            </div>
          </div>

          <div className="panel p-4">
            <h2 className="mb-3 text-[13px] font-semibold text-[var(--text-primary)]">AI Reasoning</h2>
            <p className="mb-4 text-[11px] text-[var(--text-muted)]">
              The temporal state machine confirmed this event by advancing through the full behavioral sequence.
            </p>
            <EventTimeline satisfied={reasoning} variant="checklist" />
            {isConfirmed && (
              <div className="mt-4 flex items-center gap-2 rounded-lg bg-[var(--accent)]/10 px-3 py-2">
                <CheckCircle2 className="h-4 w-4 text-[var(--accent)]" />
                <span className="text-[12px] font-semibold text-[var(--accent)]">Final Decision: Littering Confirmed</span>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ icon: Icon, label, value, mono }: { icon: typeof Camera; label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center gap-3">
      <Icon className="h-4 w-4 shrink-0 text-[var(--text-muted)]" />
      <div className="min-w-0 flex-1">
        <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">{label}</div>
        <div className={cn("truncate text-[13px] text-[var(--text-primary)]", mono && "mono")}>{value}</div>
      </div>
    </div>
  );
}
