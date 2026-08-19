import { cn } from "../lib/utils";
import { statusBadge } from "../lib/utils";

export function Badge({ status, className }: { status: string; className?: string }) {
  const b = statusBadge(status);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold",
        b.className,
        className,
      )}
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" />
      {b.label}
    </span>
  );
}
