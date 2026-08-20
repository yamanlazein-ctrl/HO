import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { FileVideo, Search, ImageOff } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getEvents, getEvidence, evidenceFileUrl } from "../lib/api";
import { Badge } from "../components/Badge";
import { formatTime } from "../lib/utils";

export function EvidencePage() {
  const [search, setSearch] = useState("");
  const { data: eventsList } = useFetch(() => getEvents(100, 0), [], 4000);

  const events = eventsList?.items ?? [];

  // Fetch evidence for each event (best-effort, sequential to avoid burst)
  const [evidenceMap, setEvidenceMap] = useState<Record<number, string>>({});
  useFetch(async () => {
    const map: Record<number, string> = {};
    for (const ev of events.slice(0, 30)) {
      try {
        const evList = await getEvidence(ev.id);
        if (evList.length > 0 && evList[0].image_path) {
          map[ev.id] = evList[0].image_path;
        }
      } catch {
        /* skip */
      }
    }
    setEvidenceMap(map);
  }, [events.length]);

  const filtered = useMemo(() => {
    if (!search) return events;
    const q = search.toLowerCase();
    return events.filter((e) => `#${e.id} ${e.object_type} ${e.status}`.toLowerCase().includes(q));
  }, [events, search]);

  return (
    <div className="mx-auto max-w-[1600px] space-y-5 p-5 lg:p-7">
      <div>
        <h1 className="text-xl font-bold text-[var(--text-primary)]">Evidence</h1>
        <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
          Snapshot & video evidence gallery — pre-event, event, post-event clips
        </p>
      </div>

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[var(--text-muted)]" />
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search evidence…"
          className="w-full rounded-lg border border-[var(--border-default)] bg-[var(--bg-elevated)] py-2 pl-9 pr-3 text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none focus:border-[var(--accent)]"
        />
      </div>

      {filtered.length === 0 && (
        <div className="panel flex flex-col items-center justify-center py-16">
          <FileVideo className="h-10 w-10 text-[var(--text-muted)]" />
          <p className="mt-3 text-[13px] text-[var(--text-secondary)]">No evidence available</p>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
        {filtered.map((ev) => {
          const snapPath = evidenceMap[ev.id];
          return (
            <Link
              key={ev.id}
              to={`/violations/${ev.id}`}
              className="panel group overflow-hidden transition-colors hover:border-[var(--accent)]/40"
            >
              <div className="relative aspect-video bg-[var(--bg-base)]">
                {snapPath ? (
                  <img
                    src={evidenceFileUrl(snapPath)}
                    alt={`Event ${ev.id} snapshot`}
                    className="h-full w-full object-cover transition-transform group-hover:scale-105"
                    onError={(e) => ((e.target as HTMLImageElement).style.opacity = "0.2")}
                  />
                ) : (
                  <div className="flex h-full items-center justify-center">
                    <ImageOff className="h-8 w-8 text-[var(--text-muted)]" />
                  </div>
                )}
                <div className="absolute left-2 top-2 rounded bg-[var(--bg-base)]/80 px-2 py-0.5 mono text-[11px] font-bold text-[var(--text-primary)] backdrop-blur">
                  #{ev.id}
                </div>
              </div>
              <div className="p-3">
                <div className="flex items-center justify-between">
                  <span className="truncate text-[12px] font-semibold text-[var(--text-primary)]">{ev.object_type}</span>
                  <Badge status={ev.status} />
                </div>
                <div className="mt-1 text-[11px] text-[var(--text-muted)]">{formatTime(ev.timestamp)}</div>
              </div>
            </Link>
          );
        })}
      </div>
    </div>
  );
}
