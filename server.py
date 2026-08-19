"""
Naval Berth Allocation — FastAPI Server  (Phase 2)
===================================================
Session-based FastAPI server for the Naval Berth Digital Twin.
Unity drives everything via plain HTTP (UnityWebRequest) — no WebSocket needed.

Architecture
------------
  Unity (UnityWebRequest)
        |
        | HTTP POST/GET
        v
  FastAPI server  (this file)
        |
        +---> NavalFinalOptimizer  (Naval_sim_em_heuristics.py)  stateful per-session

Note: MILP integration has intentionally been omitted from this server.
naval_milp_benchmark.py is not used by the live Unity digital twin (see
Table 3 in the paper) -- it is used only for the offline evaluation in
evaluate_all.py, which produces Table 5. Keeping the two separate avoids
confusion about which component backs the paper's reported results.

Endpoints
---------
  GET    /health                      Server alive + capability check
  POST   /init_scenario               Create (or re-init) a session from Unity state
  GET    /state/{session_id}          Full snapshot for a session
  POST   /step_forward                Advance one session by N ticks
  POST   /set_weather                 Inject weather event into a session
  POST   /set_weather_event           Operator-triggered weather change w/ reschedule
  POST   /set_weights                 Update operator priority weights
  POST   /set_uncertainty             Update uncertainty params (applied on next init)
  POST   /run_full                    Run solver(s) to completion
  DELETE /session/{session_id}        Remove a session from memory
  DELETE /sessions/all                Clear all sessions

Run
---
  pip install fastapi uvicorn
  python server.py
  python server.py --port 9000 --reload

Docs (auto-generated):
  http://localhost:8000/docs
"""

from __future__ import annotations

import random
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Import GA  (Naval_sim_em_heuristics.py)
# ---------------------------------------------------------------------------
try:
    from Naval_sim_em_heuristics import NavalFinalOptimizer, VESSEL_SPECS as _VESSEL_SPECS_IMPORT
    VESSEL_SPECS = _VESSEL_SPECS_IMPORT
except ImportError:
    try:
        from Naval_sim_emissions import NavalFinalOptimizer
    except ImportError:
        print("[ERROR] Could not import NavalFinalOptimizer.")
        print("        Place Naval_sim_em_heuristics.py alongside this file.")
        sys.exit(1)


# ===========================================================================
# Problem constants
# ===========================================================================

VESSEL_SPECS: Dict[str, Any] = {
    "K": {"readiness": 94, "fatigue": 8.0, "stay_range": (72,  96),  "tugs": 2,
          "cycle": 504, "count": 3,  "assigned_piers": ["P1","P2","P7","P8"]},
    "F": {"readiness": 79, "fatigue": 4.0, "stay_range": (96,  168), "tugs": 1,
          "cycle": 336, "count": 5,  "assigned_piers": ["P4","P5","P6"]},
    "L": {"readiness": 63, "fatigue": 6.0, "stay_range": (168, 168), "tugs": 2,
          "cycle": 168, "count": 4,  "assigned_piers": ["P1","P2","P7","P8"]},
    "P": {"readiness": 31, "fatigue": 1.0, "stay_range": (96,  144), "tugs": 1,
          "cycle": 240, "count": 12, "assigned_piers": ["P3","P4","P5","P6"]},
}

DEFAULT_HORIZON_H = 168
DEFAULT_SEED      = 42

_uncertainty: Dict[str, Any] = {
    "weather_prob":     0.05,
    "stay_noise_frac":  0.10,
    "arrival_jitter":   0,
    "tug_failure_prob": 0.0,
}


# ===========================================================================
# Scenario generator
# ===========================================================================

def build_ga_scenario(
    horizon_h: int = DEFAULT_HORIZON_H,
    seed: int = DEFAULT_SEED,
    vessel_counts: Dict[str, int] = None,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    scenario: List[Dict[str, Any]] = []
    for vtype, info in VESSEL_SPECS.items():
        count = (vessel_counts or {}).get(vtype, info["count"])
        for i in range(count):
            curr_h = random.randint(0, min(info["cycle"], horizon_h))
            while curr_h < horizon_h:
                stay = random.randint(*info["stay_range"])
                noise = _uncertainty["stay_noise_frac"]
                if noise > 0:
                    stay = max(1, int(stay * (1 + random.uniform(-noise, noise))))
                arr_tick = curr_h * 2
                jitter = _uncertainty["arrival_jitter"]
                if jitter > 0:
                    arr_tick = max(0, arr_tick + random.randint(-jitter, jitter))
                scenario.append({
                    "id":       f"{vtype}{i}_{curr_h}",
                    "type":     vtype,
                    "arr":      arr_tick,
                    "arr_orig": arr_tick,
                    "stay":     stay,
                })
                curr_h += info["cycle"]
    return sorted(scenario, key=lambda v: v["arr"])


# ===========================================================================
# Canonical snapshot serialiser
# ===========================================================================

def snapshot_to_dict(sim: NavalFinalOptimizer, session_id: str = "",
                     horizon_h: int = DEFAULT_HORIZON_H) -> Dict[str, Any]:
    berths: List[Dict] = []
    for pier, layers in sim.berths.items():
        for layer_idx, occupant in enumerate(layers):
            berths.append({
                "pier":        pier,
                "layer":       layer_idx,
                "vessel_id":   occupant["id"]   if occupant else None,
                "vessel_type": occupant["type"] if occupant else None,
                "occupied":    occupant is not None,
            })

    berthed_ids = {
        occ["id"]
        for layers in sim.berths.values()
        for occ in layers if occ is not None
    }

    vessels: List[Dict] = []
    for v in sim.scenario:
        if v["id"] in berthed_ids:
            status = "berthed"
        elif v.get("arr", 0) > sim.t:
            status = "queued"
        else:
            status = "departed"
        vessels.append({
            "id":       v["id"],
            "type":     v["type"],
            "status":   status,
            "arr_tick": v["arr"],
            "arr_h":    v["arr"] // 2,
            "stay_h":   v["stay"],
        })

    finished = sim.t >= horizon_h * 2

    return {
        "session_id":    session_id,
        "tick":          sim.t,
        "time_h":        round(sim.t / 2, 2),
        "weather_level": sim.weather_level,
        "weather_rem":   sim.weather_rem,
        "finished":      finished,
        "metrics": {
            "shifting":  sim.shifting,
            "fatigue":   round(sim.fatigue, 2),
            "delay":     round(sim.delay / 2, 2),
            "emissions": round(getattr(sim, "emissions", 0.0) / 1000, 2),
            "combined":  round(sim.shifting + sim.fatigue + sim.delay / 2, 2),
        },
        "berths":  berths,
        "vessels": vessels,
    }


# ===========================================================================
# Session store
# ===========================================================================

class Session:
    def __init__(self, session_id: str, scenario: List[Dict], horizon_h: int,
                 seed: int, mode: str,
                 alpha: float = 0.25, beta: float = 0.25,
                 gamma: float = 0.25, delta: float = 0.25):
        self.session_id = session_id
        self.scenario   = scenario
        self.horizon_h  = horizon_h
        self.seed       = seed
        self.mode       = mode
        self.sim        = NavalFinalOptimizer(scenario, mode=mode, record_log=False)
        self.sim.alpha  = alpha
        self.sim.beta   = beta
        self.sim.gamma  = gamma
        self.sim.delta  = delta
        self.created_at = time.time()

    def set_weights(self, alpha, beta, gamma, delta):
        self.sim.alpha = alpha
        self.sim.beta  = beta
        self.sim.gamma = gamma
        self.sim.delta = delta

    def snapshot(self) -> Dict[str, Any]:
        return snapshot_to_dict(self.sim, self.session_id, self.horizon_h)


_sessions: Dict[str, Session] = {}


def _get_session(session_id: str) -> Session:
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found. "
                                 "Call /init_scenario first.")
    return _sessions[session_id]


# ===========================================================================
# Pydantic models
# ===========================================================================

class UnityState(BaseModel):
    scenario:         List[Dict[str, Any]] = Field(default_factory=list)
    horizon_h:        int                  = Field(DEFAULT_HORIZON_H)
    seed:             int                  = Field(DEFAULT_SEED)
    mode:             str                  = Field("GA")
    vessel_counts:    Optional[Dict[str, int]] = Field(None)
    current_time:     int                  = Field(0, ge=0)
    weather_override: Optional[int]        = Field(None, ge=0, le=3)
    force_recalc:     bool                 = Field(False)
    session_id:       Optional[str]        = Field(None)


class StepRequest(BaseModel):
    session_id: str
    ticks: int = Field(1, ge=1, le=4800)


class WeatherRequest(BaseModel):
    session_id: str
    level:      int   = Field(..., ge=0, le=3)
    duration_h: float = Field(4.0, gt=0)


class WeatherEventRequest(BaseModel):
    session_id:  str
    level:       int   = Field(..., ge=0, le=3)
    duration_h:  float = Field(8.0, gt=0)


class WeightRequest(BaseModel):
    session_id: str
    alpha: float = Field(0.25, ge=0.0, le=1.0)
    beta:  float = Field(0.25, ge=0.0, le=1.0)
    gamma: float = Field(0.25, ge=0.0, le=1.0)
    delta: float = Field(0.25, ge=0.0, le=1.0)


class UncertaintyRequest(BaseModel):
    weather_prob:     Optional[float] = Field(None, ge=0.0, le=1.0)
    stay_noise_frac:  Optional[float] = Field(None, ge=0.0, le=1.0)
    arrival_jitter:   Optional[int]   = Field(None, ge=0)
    tug_failure_prob: Optional[float] = Field(None, ge=0.0, le=1.0)


class RunFullRequest(BaseModel):
    horizon_h: int                        = Field(DEFAULT_HORIZON_H)
    seed:      int                        = Field(DEFAULT_SEED)
    solver:    str                        = Field("GA")
    vessels:   Optional[List[Dict[str, Any]]] = None


# ===========================================================================
# App
# ===========================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[naval-api] Starting.")
    yield
    print("[naval-api] Shutdown.")


app = FastAPI(
    title="Naval Berth Digital Twin API",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===========================================================================
# Housekeeping
# ===========================================================================

@app.get("/health", tags=["Housekeeping"])
def health():
    return {
        "status":      "ok",
        "version":     "2.1.0",
        "sessions":    len(_sessions),
        "uncertainty": _uncertainty,
    }


@app.delete("/session/{session_id}", tags=["Housekeeping"])
def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(404, f"Session '{session_id}' not found.")
    del _sessions[session_id]


@app.delete("/sessions/all", tags=["Housekeeping"])
def delete_all_sessions():
    count = len(_sessions)
    _sessions.clear()
    return {"ok": True, "deleted": count}


# ===========================================================================
# Simulation control
# ===========================================================================

@app.post("/init_scenario", tags=["Simulation"])
def init_scenario(state: UnityState):
    sid = state.session_id or str(uuid.uuid4())

    if sid in _sessions and not state.force_recalc:
        return {"session_id": sid, "reused": True,
                "state": _sessions[sid].snapshot()}

    scenario = state.scenario if state.scenario else build_ga_scenario(
        state.horizon_h, state.seed, state.vessel_counts)

    try:
        session = Session(
            session_id=sid,
            scenario=scenario,
            horizon_h=state.horizon_h,
            seed=state.seed,
            mode=state.mode,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to create session: {e}")

    if state.current_time > 0:
        for _ in range(state.current_time):
            if session.sim.t >= session.horizon_h * 2:
                break
            session.sim.step()

    if state.weather_override is not None:
        session.sim.weather_level = state.weather_override

    _sessions[sid] = session

    from collections import Counter
    type_counts = dict(Counter(v["type"] for v in scenario))
    counts_str = "  ".join(f"{k}:{v}" for k, v in sorted(type_counts.items()))
    print(f"[naval-api] NEW SESSION {sid[:8]}... | "
          f"{len(scenario)} vessels ({counts_str}) | "
          f"horizon={state.horizon_h}h | seed={state.seed} | mode={state.mode}")

    return {
        "session_id":   sid,
        "reused":       False,
        "vessels":      len(scenario),
        "vessel_counts": type_counts,
        "horizon_h":    state.horizon_h,
        "seed":         state.seed,
        "mode":         state.mode,
        "state":        session.snapshot(),
    }


@app.get("/state/{session_id}", tags=["Simulation"])
def get_state(session_id: str):
    return _get_session(session_id).snapshot()


@app.post("/step_forward", tags=["Simulation"])
def step_forward(req: StepRequest):
    session = _get_session(req.session_id)
    if session.sim.t >= session.horizon_h * 2:
        return {"finished": True, "state": session.snapshot()}
    try:
        for _ in range(req.ticks):
            if session.sim.t >= session.horizon_h * 2:
                break
            session.sim.step()
        return {"finished": session.sim.t >= session.horizon_h * 2, "state": session.snapshot()}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/set_weather", tags=["Simulation"])
def set_weather(req: WeatherRequest):
    session = _get_session(req.session_id)
    try:
        session.sim.weather_level = req.level
        session.sim.weather_rem   = int(req.duration_h * 2)
        return {
            "ok":            True,
            "weather_level": session.sim.weather_level,
            "weather_rem":   session.sim.weather_rem,
            "duration_h":    req.duration_h,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/set_weather_event", tags=["Simulation"])
def set_weather_event(req: WeatherEventRequest):
    session = _get_session(req.session_id)
    sim     = session.sim
    t       = sim.t

    metrics_before = {
        "shifting": sim.shifting,
        "fatigue":  round(sim.fatigue, 2),
        "delay":    round(sim.delay / 2, 2),
        "combined": round(sim.shifting + sim.fatigue + sim.delay / 2, 2),
    }

    old_level = sim.weather_level

    sim.weather_level  = req.level
    sim.weather_rem    = int(req.duration_h * 2) + 1
    sim._forced_weather = True

    affected = []
    new_level = req.level

    for v in sim.scenario:
        spec         = VESSEL_SPECS.get(v["type"], {})
        weather_limit = spec.get("weather_limit", 0)

        if v["arr"] >= t:
            if new_level > weather_limit:
                delay_ticks = int(req.duration_h * 2)
                old_arr = v["arr"]
                v["arr"] = max(v["arr"], t + delay_ticks)
                if v["arr"] != old_arr:
                    affected.append({
                        "id":     v["id"],
                        "type":   v["type"],
                        "action": "arrival_delayed",
                        "from":   old_arr // 2,
                        "to":     v["arr"] // 2,
                    })

    for pier, layers in sim.berths.items():
        for l_idx, v in enumerate(layers):
            if v is None:
                continue
            spec          = VESSEL_SPECS.get(v["type"], {})
            weather_limit = spec.get("weather_limit", 0)
            if new_level > weather_limit and v.get("act_dep", 9999) <= t + 4:
                old_dep = v.get("act_dep", 0)
                v["act_dep"] = t + int(req.duration_h * 2)
                affected.append({
                    "id":     v["id"],
                    "type":   v["type"],
                    "action": "departure_delayed",
                    "pier":   pier,
                    "layer":  l_idx,
                    "from":   old_dep // 2,
                    "to":     v["act_dep"] // 2,
                })

    metrics_after = {
        "shifting": sim.shifting,
        "fatigue":  round(sim.fatigue, 2),
        "delay":    round(sim.delay / 2, 2),
        "combined": round(sim.shifting + sim.fatigue + sim.delay / 2, 2),
    }

    weather_names = {0: "Clear", 1: "Light", 2: "Moderate", 3: "Storm"}
    message = (
        f"Weather changed: {weather_names.get(old_level,'?')} → "
        f"{weather_names.get(new_level,'?')} "
        f"({len(affected)} vessels rescheduled)"
    )

    print(f"[naval-api] WEATHER EVENT | {message} | "
          f"delay {metrics_before['delay']:.1f}h → {metrics_after['delay']:.1f}h | "
          f"shifting {metrics_before['shifting']} → {metrics_after['shifting']}")
    for v in affected:
        print(f"  {v['id']} ({v['type']}): {v['action']}  "
              f"{v['from']:.0f}h → {v['to']:.0f}h")

    return {
        "ok":             True,
        "message":        message,
        "weather_level":  new_level,
        "duration_h":     req.duration_h,
        "affected":       affected,
        "metrics_before": metrics_before,
        "metrics_after":  metrics_after,
        "state":          session.snapshot(),
    }


@app.post("/set_weights", tags=["Simulation"])
def set_weights(req: WeightRequest):
    session = _get_session(req.session_id)
    total   = req.alpha + req.beta + req.gamma + req.delta
    if total <= 0:
        raise HTTPException(400, "Weights must sum to > 0")
    alpha = req.alpha / total
    beta  = req.beta  / total
    gamma = req.gamma / total
    delta = req.delta / total
    session.set_weights(alpha, beta, gamma, delta)
    return {
        "ok":    True,
        "alpha": round(alpha, 4),
        "beta":  round(beta,  4),
        "gamma": round(gamma, 4),
        "delta": round(delta, 4),
    }


@app.post("/set_uncertainty", tags=["Simulation"])
def set_uncertainty(req: UncertaintyRequest):
    updates = req.model_dump(exclude_none=True)
    _uncertainty.update(updates)
    return {"ok": True, "uncertainty": _uncertainty}


# ===========================================================================
# Solvers  (non-MILP)
# ===========================================================================

@app.post("/run_full", tags=["Solvers"])
def run_full(req: RunFullRequest):
    scenario = req.vessels or build_ga_scenario(req.horizon_h, req.seed)

    def _run(mode: str) -> Dict[str, Any]:
        sim = NavalFinalOptimizer(scenario, mode=mode, record_log=False)
        t0  = time.time()
        res = sim.run(max_h=req.horizon_h)
        return {
            "mode":      mode,
            "elapsed_s": round(time.time() - t0, 3),
            "shifting":  res["shifting"],
            "fatigue":   round(res["fatigue"], 2),
            "delay":     round(res["delay"] / 2, 2),
            "combined":  round(res["shifting"] + res["fatigue"] + res["delay"] / 2, 2),
        }

    try:
        if req.solver == "BOTH":
            return {"horizon_h": req.horizon_h, "seed": req.seed,
                    "results": {"GA": _run("GA"), "FCFS": _run("FCFS")}}
        return {"horizon_h": req.horizon_h, "seed": req.seed,
                "results": _run(req.solver)}
    except Exception as e:
        raise HTTPException(500, str(e))


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Naval Berth Digital Twin API")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Naval Berth Digital Twin API  v2.1.0")
    print(f"  http://localhost:{args.port}")
    print(f"  Docs:  http://localhost:{args.port}/docs")
    print(f"{'='*60}\n")

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )