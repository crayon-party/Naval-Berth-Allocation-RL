"""
naval_berth_env.py
==================
Gymnasium environment for the naval berth allocation problem, wrapping the
RT-CMOS simulator core (adapted from Naval_sim_core / NavalFinalOptimizer).

MDP design
----------
* A decision point occurs each time a vessel is ready to berth
  (arr == t, tugs available, weather within limit).
* Action space: Discrete(20) = 19 (pier, layer) slots + WAIT.
    slot index order: P1L0,P1L1,P1L2, P2L0..2, P3L0, P4L0..1, P5L0..1,
                      P6L0..1, P7L0..2, P8L0..2   (19 slots), action 19 = WAIT
* Invalid slots (occupied / wrong pier for type / adjacent incompatibility)
  are masked out.  WAIT is always valid.
* Reward: negative increment of  w_s*shifting + w_f*fatigue + w_d*delay(h),
  divided by REWARD_SCALE.  Unserved vessels at the horizon incur a terminal
  penalty so the agent cannot WAIT forever.
* Time: simulator ticks are 0.5 h (as in RT-CMOS).  Scenario generation is in
  1 h units (identical to the MILP benchmark script) and converted to ticks,
  so MILP / greedy / DRL all see the *same instances*.
Comparability notes (matters for the paper)
-------------------------------------------
* The MILP counts fatigue at both the berthing (start) and departure events,
  matching the simulator's full fatigue model. The env still tracks
  `fatigue_arr` and `fatigue_dep` separately so the evaluation script can
  compute a berthing-only objective (shifting + fatigue_arr + delay) for
  diagnostic decomposition purposes, as well as the full operational
  objective (shifting + fatigue_arr + fatigue_dep + delay), which is now
  the MILP-comparable one.
"""""

import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Problem constants (identical to RT-CMOS / MILP scripts) ──────────────────

VESSEL_SPECS = {
    'K': {'readiness': 94, 'fatigue': 8.0, 'stay_range': (72, 96),  'tugs': 2, 'duration': 2,
          'cycle': 504, 'count': 3,  'assigned_piers': ['P1', 'P2', 'P7', 'P8'], 'weather_limit': 2},
    'F': {'readiness': 79, 'fatigue': 4.0, 'stay_range': (96, 168), 'tugs': 1, 'duration': 1,
          'cycle': 336, 'count': 5,  'assigned_piers': ['P4', 'P5', 'P6'],       'weather_limit': 1},
    'L': {'readiness': 63, 'fatigue': 6.0, 'stay_range': (168, 168),'tugs': 2, 'duration': 2,
          'cycle': 168, 'count': 4,  'assigned_piers': ['P1', 'P2', 'P7', 'P8'], 'weather_limit': 1},
    'P': {'readiness': 31, 'fatigue': 1.0, 'stay_range': (96, 144), 'tugs': 1, 'duration': 1,
          'cycle': 240, 'count': 12, 'assigned_piers': ['P3', 'P4', 'P5', 'P6'], 'weather_limit': 0},
}

PIER_CONFIG = {
    'P1': {'layers': 3}, 'P2': {'layers': 3}, 'P3': {'layers': 1},
    'P4': {'layers': 2}, 'P5': {'layers': 2}, 'P6': {'layers': 2},
    'P7': {'layers': 3}, 'P8': {'layers': 3},
}

INCOMPATIBLE = [frozenset({'K', 'P'}), frozenset({'P', 'L'})]
N_TUGS = 6
NIGHT_MULT = 10 # night fatigue multiplier (matches sim)
VESSEL_TYPES = ['K', 'F', 'L', 'P']

# Fixed slot enumeration
SLOTS = [(p, l) for p in PIER_CONFIG for l in range(PIER_CONFIG[p]['layers'])]
N_SLOTS = len(SLOTS)              # 19
WAIT_ACTION = N_SLOTS             # 19
N_ACTIONS = N_SLOTS + 1           # 20

REWARD_SCALE = 10.0
UNSERVED_PENALTY = 100.0          # per vessel never berthed (pre-scaling)


# ── Scenario generator (identical logic to MILP benchmark; 1 h units) ────────

def generate_scenario(horizon_h=72, seed=42):
    rng = random.Random(seed)
    out = []
    for vtype, info in VESSEL_SPECS.items():
        for i in range(info['count']):
            curr = rng.randint(0, min(info['cycle'], horizon_h))
            while curr < horizon_h:
                stay = rng.randint(*info['stay_range'])
                if stay <= horizon_h - curr:
                    out.append({'id': f"{vtype}{i}_{curr}", 'type': vtype,
                                'arr': curr, 'stay': stay})
                curr += info['cycle']
    return sorted(out, key=lambda v: v['arr'])


# ── Environment ───────────────────────────────────────────────────────────────

class NavalBerthEnv(gym.Env):
    """One decision = berth-or-wait for the vessel currently requesting."""

    metadata = {"render_modes": []}

    def __init__(self, horizon_h=240, scenario_seed=None,
                 weather_prob=0.0, lunch_break=False,
                 w_shift=1.0, w_fatigue=1.0, w_delay=1.0,
                 queue_rule="wait"):
        super().__init__()
        self.horizon_h = horizon_h
        self.T_ticks = horizon_h * 2
        self.scenario_seed = scenario_seed     # None -> random each reset
        self.weather_prob = weather_prob
        self.lunch_break = lunch_break
        self.w = (w_shift, w_fatigue, w_delay)
        self.queue_rule = queue_rule        # next to self.w = (...)

        # obs: global(5) + vessel(8) + per-slot(19*4) = 89
        self.obs_dim = 5 + 8 + N_SLOTS * 4
        self.observation_space = spaces.Box(-1.0, 2.0, (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self._rng = random.Random()

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def _is_night(t):
        return (t % 48) >= 44 or (t % 48) <= 14

    def _fatigue(self, vtype, t):
        return VESSEL_SPECS[vtype]['fatigue'] * (NIGHT_MULT if self._is_night(t) else 1)

    def _compatible(self, t1, t2):
        return t1 == t2 or frozenset({t1, t2}) not in INCOMPATIBLE

    def _cost(self):
        ws, wf, wd = self.w
        return (ws * self.shifting
                + wf * (self.fatigue_arr + self.fatigue_dep)
                + wd * self.delay)          # delay tracked in hours

    # ── episode setup ─────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng.seed(seed)
        sc_seed = (self.scenario_seed if self.scenario_seed is not None
                   else self._rng.randint(0, 10**9))
        hours = generate_scenario(self.horizon_h, seed=sc_seed)
        self.scenario = [{'id': v['id'], 'type': v['type'],
                          'arr': v['arr'] * 2, 'arr_orig': v['arr'] * 2,
                          'stay': v['stay'], 'berthed': False}
                         for v in hours]
        self.t = 0
        self.berths = {p: [None] * PIER_CONFIG[p]['layers'] for p in PIER_CONFIG}
        self.tug_free_time = [0] * N_TUGS
        self.shifting = 0
        self.fatigue_arr = 0.0
        self.fatigue_dep = 0.0
        self.delay = 0.0                     # hours
        self.weather_level = 0
        self.weather_rem = 0
        self._tick_queue = []                # vessels awaiting decision this tick
        self._avail_tugs = 0
        self._current = None
        self._prev_cost = 0.0
        self._advance()
        return self._obs(), {"scenario_seed": sc_seed,
                             "n_vessels": len(self.scenario)}

    # ── simulator advance until next decision point ───────────────────────────
    def _update_weather(self):
        if self.weather_rem <= 0:
            if self.weather_prob > 0 and self._rng.random() < self.weather_prob:
                self.weather_level = self._rng.choice([1, 2, 3])
                self.weather_rem = self._rng.randint(2, 8)
            else:
                self.weather_level = 0
        else:
            self.weather_rem -= 1

    def _process_departures(self):
        t = self.t
        out = [(p, i, v) for p, layers in self.berths.items()
               for i, v in enumerate(layers) if v and v['act_dep'] <= t]
        for p, i, v in sorted(out, key=lambda x: VESSEL_SPECS[x[2]['type']]['readiness'],
                              reverse=True):
            spec = VESSEL_SPECS[v['type']]
            if self.weather_level <= spec['weather_limit'] and self._avail_tugs >= spec['tugs']:
                blockers = [self.berths[p][j] for j in range(i + 1, len(self.berths[p]))
                            if self.berths[p][j]]
                if not blockers:
                    assigned = 0
                    for tid in range(N_TUGS):
                        if self.tug_free_time[tid] <= t and assigned < spec['tugs']:
                            self.tug_free_time[tid] = t + spec['duration']
                            assigned += 1
                    self.fatigue_dep += self._fatigue(v['type'], t)
                    self.berths[p][i] = None
                    self._avail_tugs -= spec['tugs']
                else:
                    self.shifting += 1
                    v['act_dep'] += 1
                    self.delay += 0.5
            else:
                v['act_dep'] += 1
                self.delay += 0.5

    def _begin_tick(self):
        self._update_weather()
        lunch = self.lunch_break and (self.t % 48) in (24, 25)
        self._avail_tugs = (0 if (lunch or self.weather_level == 3)
                            else sum(1 for f in self.tug_free_time if f <= self.t))
        self._process_departures()
        ready = [v for v in self.scenario if not v['berthed'] and v['arr'] == self.t]
        # longest-waiting first (neutral ordering shared by all policies)
        rule = self.queue_rule
        if rule == "edd":
            key = lambda v: v['arr_orig'] + v['stay'] * 2  # earliest due first
        elif rule == "spt":
            key = lambda v: v['stay']  # shortest stay first
        elif rule == "urgent":
            key = lambda v: -VESSEL_SPECS[v['type']]['readiness']  # highest readiness first
        else:
            key = lambda v: -(self.t - v['arr_orig'])  # longest waiting first
        self._tick_queue = sorted(ready, key=key)

    def _next_decision_from_queue(self):
        """Pop queue; auto-defer vessels blocked by tugs/weather (no choice exists)."""
        while self._tick_queue:
            v = self._tick_queue.pop(0)
            spec = VESSEL_SPECS[v['type']]
            if self._avail_tugs >= spec['tugs'] and self.weather_level <= spec['weather_limit']:
                self._current = v
                return True
            v['arr'] += 1
            self.delay += 0.5
        return False

    def _advance(self):
        """Advance ticks until a decision point or horizon end."""
        self._current = None
        if self._tick_queue and self._next_decision_from_queue():
            return
        while self.t < self.T_ticks:
            if not self._tick_queue:
                self._begin_tick()
            if self._next_decision_from_queue():
                return
            self.t += 1
        # drain remaining departures at horizon (cost bookkeeping only)

    # ── masking ───────────────────────────────────────────────────────────────
    MAX_WAIT_TICKS = 48  # 24 h: WAIT forbidden beyond this if any berth is feasible

    def action_masks(self):
        mask = np.zeros(N_ACTIONS, dtype=bool)
        v = self._current
        if v is None:
            mask[WAIT_ACTION] = True
            return mask
        spec = VESSEL_SPECS[v['type']]
        any_slot = False
        for k, (p, l) in enumerate(SLOTS):
            if p not in spec['assigned_piers']:
                continue
            if self.berths[p][l] is not None:
                continue
            ok = True
            for i, other in enumerate(self.berths[p]):
                if other and abs(i - l) == 1 and not self._compatible(v['type'], other['type']):
                    ok = False
                    break
            if ok:
                mask[k] = True
                any_slot = True
        waited = self.t - v['arr_orig']
        mask[WAIT_ACTION] = (not any_slot) or (waited < self.MAX_WAIT_TICKS)
        return mask

    # ── observation ───────────────────────────────────────────────────────────
    def _obs(self):
        o = np.zeros(self.obs_dim, dtype=np.float32)
        t = self.t
        o[0] = t / self.T_ticks
        o[1] = 1.0 if self._is_night(t) else 0.0
        o[2] = self.weather_level / 3.0
        o[3] = self._avail_tugs / N_TUGS
        o[4] = sum(1 for v in self.scenario
                   if not v['berthed'] and v['arr'] <= t) / 10.0
        v = self._current
        if v is not None:
            ti = VESSEL_TYPES.index(v['type'])
            o[5 + ti] = 1.0
            spec = VESSEL_SPECS[v['type']]
            o[9] = spec['tugs'] / 2.0
            o[10] = v['stay'] / 168.0
            o[11] = (t - v['arr_orig']) / 48.0
            o[12] = spec['fatigue'] / 8.0
            my_dep = t + v['stay'] * 2
            for k, (p, l) in enumerate(SLOTS):
                base = 13 + k * 4
                occ = self.berths[p][l]
                o[base] = 1.0 if occ is not None else 0.0
                if occ is not None:
                    o[base + 1] = max(0.0, (occ['act_dep'] - t)) / 336.0
                # blocking indicator if WE park at (p,l)
                blk = 0.0
                for i, other in enumerate(self.berths[p]):
                    if other and ((l < i and my_dep < other['dep_t'])
                                  or (l > i and my_dep > other['dep_t'])):
                        blk = 1.0
                o[base + 2] = blk
                # incompatibility with any adjacent occupant
                inc = 0.0
                for i, other in enumerate(self.berths[p]):
                    if other and abs(i - l) == 1 and not self._compatible(v['type'], other['type']):
                        inc = 1.0
                o[base + 3] = inc
        return o

    # ── step ──────────────────────────────────────────────────────────────────
    def step(self, action):
        v = self._current
        assert v is not None, "step() called with no pending decision"
        t = self.t
        if action == WAIT_ACTION or not self.action_masks()[action]:
            v['arr'] += 1
            self.delay += 0.5
        else:
            p, l = SLOTS[action]
            spec = VESSEL_SPECS[v['type']]
            assigned = 0
            for tid in range(N_TUGS):
                if self.tug_free_time[tid] <= t and assigned < spec['tugs']:
                    self.tug_free_time[tid] = t + spec['duration']
                    assigned += 1
            self._avail_tugs -= spec['tugs']
            v['act_dep'] = t + v['stay'] * 2
            v['dep_t'] = v['act_dep']
            v['berthed'] = True
            self.berths[p][l] = v
            self.fatigue_arr += self._fatigue(v['type'], t)

        self._advance()
        terminated = self._current is None
        cost = self._cost()
        reward = -(cost - self._prev_cost) / REWARD_SCALE
        self._prev_cost = cost
        info = {}
        if terminated:
            unserved = sum(1 for u in self.scenario if not u['berthed'])
            reward -= unserved * UNSERVED_PENALTY / REWARD_SCALE
            info = self.metrics(unserved=unserved)
        return self._obs(), reward, terminated, False, info

    def metrics(self, unserved=None):
        if unserved is None:
            unserved = sum(1 for u in self.scenario if not u['berthed'])
        return {'shifting': self.shifting,
                'fatigue_arr': round(self.fatigue_arr, 1),
                'fatigue_total': round(self.fatigue_arr + self.fatigue_dep, 1),
                'delay_h': round(self.delay, 1),
                'unserved': unserved,
                'obj_milp_comparable': round(self.shifting + self.fatigue_arr + self.delay, 1),
                'obj_full': round(self._cost(), 1)}


# ── Baseline policies (operate on the SAME env -> identical dynamics) ─────────

F_REF, D_REF, E_REF, SCALE = 400.0, 600.0, 2_000_000.0, 50_000.0
E_IDLE = {'K': 2134, 'F': 1280, 'L': 2560, 'P': 427}


class GreedyPolicy:
    """Replicates the final RT-CMOS greedy constructive multi-objective
    scheduler (normalized fitness with fatigue / wait / emissions terms,
    blocking penalty, and multi-criteria night-deferral rule)."""

    def __init__(self, alpha=0.25, beta=0.25, gamma=0.25, delta=0.25,
                 night_defer=True, max_night_defer_ticks=6):
        self.beta, self.gamma, self.delta = beta, gamma, delta
        self.night_defer = night_defer
        self.cap = max_night_defer_ticks

    def __call__(self, env: NavalBerthEnv):
        v = env._current
        t = env.t
        mask = env.action_masks()
        wait_ticks = t - v['arr_orig']
        e_idle = E_IDLE[v['type']]

        if self.night_defer and env._is_night(t):
            f_cost = self.beta * (env._fatigue(v['type'], t) / F_REF)
            accum = (self.gamma * (wait_ticks / D_REF)
                     + self.delta * (e_idle * wait_ticks * 0.5 / E_REF))
            if f_cost > accum and wait_ticks < self.cap:
                return WAIT_ACTION

        best_s, best_a = float('inf'), WAIT_ACTION
        my_dep = t + v['stay'] * 2
        wait_h = wait_ticks * 0.5
        curr_f = env._fatigue(v['type'], t)
        for k, (p, l) in enumerate(SLOTS):
            if not mask[k]:
                continue
            pen = (self.beta * (curr_f / F_REF) * SCALE
                   - self.gamma * (wait_ticks / D_REF) * SCALE
                   - self.delta * (e_idle * wait_h / E_REF) * SCALE)
            for i, other in enumerate(env.berths[p]):
                if other and ((l < i and my_dep < other['dep_t'])
                              or (l > i and my_dep > other['dep_t'])):
                    pen += SCALE
            if pen < best_s:
                best_s, best_a = pen, k
        return best_a


class FCFSPolicy:
    """First feasible slot by pier/layer index; never waits voluntarily."""
    def __call__(self, env: NavalBerthEnv):
        mask = env.action_masks()
        for k in range(N_SLOTS):
            if mask[k]:
                return k
        return WAIT_ACTION


class RandomPolicy:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
    def __call__(self, env: NavalBerthEnv):
        mask = env.action_masks()
        return int(self.rng.choice(np.flatnonzero(mask)))


def rollout(env: NavalBerthEnv, policy, seed=0):
    obs, info = env.reset(seed=seed)
    done = info.get('n_vessels', 1) == 0
    final = env.metrics() if done else None
    while not done:
        a = policy(env)
        obs, r, done, _, info = env.step(a)
        if done:
            final = info
    return final


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'seed':<6}{'policy':<9}{'shift':>6}{'fat(arr)':>9}{'delay_h':>8}"
          f"{'unserved':>9}{'obj(MILP-cmp)':>14}")
    for seed in (42, 7, 13):
        for name, pol in (('FCFS', FCFSPolicy()),
                          ('Greedy', GreedyPolicy()),
                          ('Random', RandomPolicy(seed))):
            env = NavalBerthEnv(horizon_h=240, scenario_seed=seed,
                                weather_prob=0.0, lunch_break=False)
            m = rollout(env, pol, seed=seed)
            print(f"{seed:<6}{name:<9}{m['shifting']:>6}{m['fatigue_arr']:>9.1f}"
                  f"{m['delay_h']:>8.1f}{m['unserved']:>9}"
                  f"{m['obj_milp_comparable']:>14.1f}")
