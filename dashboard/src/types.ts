// Domain types matching the FastAPI backend schemas (backend/schemas.py)
// and the new status endpoint (backend/routers/status.py).

export interface Camera {
  id: number;
  name: string;
  location: string;
  status: string;
  created_at: string;
}

export interface Event {
  id: number;
  camera_id: number;
  person_track_id: string | null;
  object_track_id: string | null;
  object_type: string;
  timestamp: string;
  confidence: number;
  status: EventStatus;
  created_at: string;
}

export interface EventList {
  items: Event[];
  total: number;
  limit: number;
  offset: number;
}

export interface Evidence {
  id: number;
  event_id: number;
  image_path: string | null;
  video_path: string | null;
  duration_sec: number | null;
  created_at: string;
}

export interface Statistics {
  total_events: number;
  events_today: number;
  per_object_type: Record<string, number>;
  avg_confidence: number;
}

export type EventStatus = "new" | "reviewing" | "confirmed" | "rejected";

// /api/status — the live system status bar payload
export interface SystemStatus {
  system_online: boolean;
  ai_engine: {
    status: "online" | "offline" | "degraded";
    model_loaded: boolean;
    classes: string[];
  };
  camera: {
    status: "online" | "offline" | "waiting";
    fps: number | null;
    resolution: string | null;
    source: string | null;
  };
  processing: {
    fps: number | null;
    latency_ms: number | null;
    analysis_fps: number | null;
  };
  buffer: {
    window_seconds: number;
    frames_buffered: number;
    buffer_duration: number;
  };
  live_state?: {
    ai_state: string;
    active_pairs: number;
    entities: Array<{
      trackId: number;
      label: string;
      bbox: { x: number; y: number; w: number; h: number };
      confidence: number;
      isPerson: boolean;
    }>;
  };
  events_today: number;
  active_cameras: number;
  updated_at: string;
}

// AI state shown on live monitoring — mirrors LitterState enum
export type LitterState =
  | "UNKNOWN"
  | "INTERACTING"
  | "HOLDING"
  | "RELEASE"
  | "OBJECT_ON_GROUND"
  | "PERSON_AWAY"
  | "SUSPICIOUS"
  | "LITTERING_CONFIRMED"
  | "NORMAL";

// The AI reasoning checklist for the event detail page
export interface AIReasoningStep {
  label: string;
  satisfied: boolean;
}
