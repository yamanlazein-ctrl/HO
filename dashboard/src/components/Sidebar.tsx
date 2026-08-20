import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Camera,
  MonitorPlay,
  AlertTriangle,
  FileVideo,
  BarChart3,
  Settings,
  ShieldCheck,
} from "lucide-react";
import { cn } from "../lib/utils";

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
}
interface NavGroup {
  title: string;
  items: NavItem[];
}

const GROUPS: NavGroup[] = [
  {
    title: "Monitoring",
    items: [
      { to: "/", label: "Dashboard", icon: LayoutDashboard },
      { to: "/cameras", label: "Cameras", icon: Camera },
      { to: "/live", label: "Live Monitoring", icon: MonitorPlay },
      { to: "/analysis", label: "Video Analysis", icon: FileVideo },
    ],
  },
  {
    title: "Operations",
    items: [
      { to: "/violations", label: "Violations", icon: AlertTriangle },
      { to: "/evidence", label: "Evidence", icon: FileVideo },
    ],
  },
  {
    title: "System",
    items: [
      { to: "/analytics", label: "Analytics", icon: BarChart3 },
      { to: "/settings", label: "Settings", icon: Settings },
    ],
  },
];

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full w-[244px] flex-col border-r border-[var(--border-subtle)] bg-[var(--bg-panel)]">
      <div className="flex h-16 items-center gap-2.5 border-b border-[var(--border-subtle)] px-5">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[var(--accent)]/15">
          <ShieldCheck className="h-5 w-5 text-[var(--accent)]" />
        </div>
        <div className="leading-tight">
          <div className="text-[13px] font-bold tracking-wide text-[var(--text-primary)]">
            LITTERING AI
          </div>
          <div className="text-[10px] font-medium uppercase tracking-[0.18em] text-[var(--text-muted)]">
            Evidence System
          </div>
        </div>
      </div>

      <nav className="flex-1 overflow-y-auto px-3 py-4">
        {GROUPS.map((g) => (
          <div key={g.title} className="mb-5">
            <div className="px-3 pb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              {g.title}
            </div>
            <div className="space-y-0.5">
              {g.items.map((item) => (
                <NavLink
                  key={item.to}
                  to={item.to}
                  end={item.to === "/"}
                  onClick={onNavigate}
                  className={({ isActive }) =>
                    cn(
                      "flex items-center gap-3 rounded-lg px-3 py-2 text-[13px] font-medium transition-colors",
                      isActive
                        ? "bg-[var(--accent)]/12 text-[var(--accent)]"
                        : "text-[var(--text-secondary)] hover:bg-[var(--bg-hover)] hover:text-[var(--text-primary)]",
                    )
                  }
                >
                  <item.icon className="h-[18px] w-[18px] shrink-0" />
                  <span>{item.label}</span>
                </NavLink>
              ))}
            </div>
          </div>
        ))}
      </nav>

      <div className="border-t border-[var(--border-subtle)] px-5 py-3 text-[10px] text-[var(--text-muted)]">
        v1.0 · AI Video Analytics
      </div>
    </div>
  );
}
