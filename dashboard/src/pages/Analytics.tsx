import { useMemo } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";
import { BarChart3, TrendingUp, Clock, Gauge, Activity } from "lucide-react";
import { useFetch } from "../lib/useFetch";
import { getEvents, getStatistics } from "../lib/api";
import { StatCard } from "../components/StatCard";

const PIE_COLORS = ["#00e5b8", "#3b82f6", "#ffa726", "#ff4757", "#8ea0bc", "#5d6f8e"];

export function Analytics() {
  const { data: stats } = useFetch(getStatistics, []);
  const { data: eventsList } = useFetch(() => getEvents(500, 0), []);

  const events = eventsList?.items ?? [];

  // Derive real charts from actual event data — NO fabricated numbers.
  const byObjectType = useMemo(() => {
    const counts: Record<string, number> = stats?.per_object_type ?? {};
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [stats]);

  const byHour = useMemo(() => {
    const hours = Array.from({ length: 24 }, (_, h) => ({ hour: `${h}:00`, count: 0 }));
    for (const e of events) {
      const h = new Date(e.timestamp).getHours();
      if (!Number.isNaN(h)) hours[h].count += 1;
    }
    return hours;
  }, [events]);

  const byStatus = useMemo(() => {
    const counts: Record<string, number> = {};
    for (const e of events) counts[e.status] = (counts[e.status] ?? 0) + 1;
    return Object.entries(counts).map(([name, value]) => ({ name, value }));
  }, [events]);

  const confidenceBuckets = useMemo(() => {
    const buckets = [
      { range: "0–50%", count: 0 },
      { range: "50–70%", count: 0 },
      { range: "70–85%", count: 0 },
      { range: "85–100%", count: 0 },
    ];
    for (const e of events) {
      const c = e.confidence;
      if (c < 0.5) buckets[0].count += 1;
      else if (c < 0.7) buckets[1].count += 1;
      else if (c < 0.85) buckets[2].count += 1;
      else buckets[3].count += 1;
    }
    return buckets;
  }, [events]);

  const hasData = events.length > 0;

  return (
    <div className="mx-auto max-w-[1600px] space-y-6 p-5 lg:p-7">
      <div>
        <h1 className="text-xl font-bold text-[var(--text-primary)]">Analytics</h1>
        <p className="mt-0.5 text-[13px] text-[var(--text-secondary)]">
          Detection analytics derived from real recorded events — no fabricated statistics.
        </p>
      </div>

      {/* KPI row */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Total Events" value={stats?.total_events ?? 0} icon={TrendingUp} accent="info" loading={!stats} />
        <StatCard label="Events Today" value={stats?.events_today ?? 0} icon={Activity} accent="danger" loading={!stats} />
        <StatCard label="Avg Confidence" value={stats?.avg_confidence ? `${Math.round(stats.avg_confidence * 100)}%` : "—"} icon={Gauge} accent="warning" loading={!stats} />
        <StatCard label="Object Types" value={byObjectType.length} icon={BarChart3} loading={!stats} />
      </div>

      {!hasData && (
        <div className="panel flex flex-col items-center justify-center py-20">
          <BarChart3 className="h-10 w-10 text-[var(--text-muted)]" />
          <p className="mt-3 text-[13px] text-[var(--text-secondary)]">No data available</p>
          <p className="mt-1 text-[11px] text-[var(--text-muted)]">
            Analytics populate from recorded events. Run the pipeline to generate data.
          </p>
        </div>
      )}

      {hasData && (
        <div className="grid gap-6 lg:grid-cols-2">
          <ChartCard title="Violations by Object Type" icon={BarChart3}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={byObjectType}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2f4a" />
                <XAxis dataKey="name" stroke="#8ea0bc" fontSize={11} />
                <YAxis stroke="#8ea0bc" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="value" fill="#00e5b8" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Violations by Hour of Day" icon={Clock}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={byHour}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2f4a" />
                <XAxis dataKey="hour" stroke="#8ea0bc" fontSize={10} interval={2} />
                <YAxis stroke="#8ea0bc" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Event Status Distribution" icon={Activity}>
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie data={byStatus} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} innerRadius={45}>
                  {byStatus.map((_, i) => <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />)}
                </Pie>
                <Legend wrapperStyle={{ fontSize: 11, color: "#8ea0bc" }} />
                <Tooltip contentStyle={tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Confidence Distribution" icon={Gauge}>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={confidenceBuckets}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2f4a" />
                <XAxis dataKey="range" stroke="#8ea0bc" fontSize={11} />
                <YAxis stroke="#8ea0bc" fontSize={11} allowDecimals={false} />
                <Tooltip contentStyle={tooltipStyle} />
                <Bar dataKey="count" fill="#ffa726" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>
      )}
    </div>
  );
}

const tooltipStyle = {
  background: "#16233a",
  border: "1px solid #283d5c",
  borderRadius: 8,
  fontSize: 12,
  color: "#e8eef7",
};

function ChartCard({ title, icon: Icon, children }: { title: string; icon: typeof BarChart3; children: React.ReactNode }) {
  return (
    <div className="panel p-4">
      <div className="mb-4 flex items-center gap-2">
        <Icon className="h-4 w-4 text-[var(--accent)]" />
        <h2 className="text-[13px] font-semibold text-[var(--text-primary)]">{title}</h2>
      </div>
      {children}
    </div>
  );
}
