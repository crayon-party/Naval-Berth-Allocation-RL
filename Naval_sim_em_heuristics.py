import random
import numpy as np
import pandas as pd
from collections import Counter

# Normalisation reference values (based on empirical FCFS maximums, 10-seed study)
S_REF = 300.0  # shifting
F_REF = 400.0  # fatigue
D_REF = 600.0  # delay (half-hours)
E_REF = 2000000.0  # emissions (kg CO2) — updated to reflect shifting underway emissions
SCALE = 50000.0  # fitness function anchor = blocking penalty

VESSEL_SPECS = {
    # e_idle: kg CO2 per hour waiting at anchorage (proxy from Giachetti & Moore 2024,
    # mapped to Korean vessel types by closest US Navy equivalent)
    # Fatigue factors scaled x0.50 (K=4, F=2, L=3, P=0.5)
    # e_underway: kg CO2/h when underway (shifting manoeuvre) ~ e_idle x 4.5
    # K highest due to gas turbine propulsion; L lower than K despite larger displacement
    'K': {'readiness': 94, 'fatigue': 4.0, 'stay_range': (72, 96), 'tugs': 2, 'duration': 2, 'cycle': 504, 'count': 3,
          'assigned_piers': ['P1', 'P2', 'P7', 'P8'], 'weather_limit': 2, 'e_idle': 2134, 'e_underway': 12000},
    'F': {'readiness': 79, 'fatigue': 2.0, 'stay_range': (96, 168), 'tugs': 1, 'duration': 1, 'cycle': 336, 'count': 5,
          'assigned_piers': ['P4', 'P5', 'P6'], 'weather_limit': 1, 'e_idle': 1280, 'e_underway': 5760},
    'L': {'readiness': 63, 'fatigue': 3.0, 'stay_range': (168, 168), 'tugs': 2, 'duration': 2, 'cycle': 168, 'count': 4,
          'assigned_piers': ['P1', 'P2', 'P7', 'P8'], 'weather_limit': 1, 'e_idle': 2560, 'e_underway': 9000},
    'P': {'readiness': 31, 'fatigue': 0.5, 'stay_range': (96, 144), 'tugs': 1, 'duration': 1, 'cycle': 240, 'count': 12,
          'assigned_piers': ['P3', 'P4', 'P5', 'P6'], 'weather_limit': 0, 'e_idle': 427, 'e_underway': 1920}
}

PIER_CONFIG = {
    'P1': {'layers': 3}, 'P2': {'layers': 3}, 'P3': {'layers': 1},
    'P4': {'layers': 2}, 'P5': {'layers': 2}, 'P6': {'layers': 2},
    'P7': {'layers': 3}, 'P8': {'layers': 3}
}


class NavalFinalOptimizer:
    def __init__(self, scenario, mode='GA', record_log=True):
        self.initial_scenario = [v.copy() for v in scenario]
        self.mode = mode
        self.record_log = record_log
        self.reset()

    def reset(self):
        self.t = 0
        self.berths = {p: [None] * PIER_CONFIG[p]['layers'] for p in PIER_CONFIG}
        self.scenario = [v.copy() for v in self.initial_scenario]
        self.tug_free_time = [0] * 6
        self.shifting, self.fatigue, self.delay, self.emissions = 0, 0, 0, 0.0
        self.vessel_history = []
        self.weather_level = 0
        self.weather_rem = 0
        self.counts = Counter()
        self.last_move_dict = {}
        # Preserve operator weights across reset (default equal weights)
        if not hasattr(self, 'alpha'): self.alpha = 0.25
        if not hasattr(self, 'beta'):  self.beta = 0.25
        if not hasattr(self, 'gamma'): self.gamma = 0.25
        if not hasattr(self, 'delta'): self.delta = 0.25
        # GA-optimisable fitness parameters
        if not hasattr(self, 'lambda_beta'):  self.lambda_beta = 1.0
        if not hasattr(self, 'lambda_gamma'): self.lambda_gamma = 1.0
        if not hasattr(self, 'lambda_delta'): self.lambda_delta = 1.0
        if not hasattr(self, 'lambda_block'): self.lambda_block = 1.0

    def update_weather(self, t):
        if self.weather_rem <= 0:
            if random.random() < 0.05:
                self.weather_level = random.choice([1, 2, 3])
                self.weather_rem = random.randint(2, 8)
            else:
                self.weather_level = 0
        else:
            self.weather_rem -= 1
        return self.weather_level

    def is_night(self, t):
        return (t % 48) >= 44 or (t % 48) <= 14

    def check_compatibility(self, type1, type2):
        if type1 == type2: return True
        pair = {type1, type2}
        if pair == {'K', 'P'} or pair == {'P', 'L'}: return False
        return True

    def calculate_vessel_fatigue(self, vessel_id, v_type, t, is_shifting=False):
        spec_f = VESSEL_SPECS[v_type]['fatigue']
        multiplier = 10 if self.is_night(t) else 1
        total_f = spec_f * multiplier
        if is_shifting: total_f *= 1.5
        return total_f

    def evaluate_fitness(self, v, p_id, l_idx, t, wait_time):
        # Hard constraints — apply to ALL modes
        if self.berths[p_id][l_idx] is not None: return 1e15
        if p_id not in VESSEL_SPECS[v['type']]['assigned_piers']: return 1e15
        for i, other in enumerate(self.berths[p_id]):
            if other and abs(i - l_idx) == 1:
                if not self.check_compatibility(v['type'], other['type']): return 2e15

        # Non-GA modes: vessel ordering handled by vessel_priority sort above
        # Just pick first available valid slot (lowest pier+layer index)
        if self.mode in ('FCFS', 'EDD', 'SPT', 'URGENT'):
            return int(p_id[1:]) * 10 + l_idx

        alpha = getattr(self, 'alpha', 0.25)
        beta = getattr(self, 'beta', 0.25)
        gamma = getattr(self, 'gamma', 0.25)
        delta = getattr(self, 'delta', 0.25)

        penalty = 0
        curr_f = self.calculate_vessel_fatigue(v['id'], v['type'], t)
        wait_h = wait_time * 0.5  # ticks → hours
        e_idle = VESSEL_SPECS[v['type']]['e_idle']

        l_b = getattr(self, 'lambda_beta', 1.0)
        l_g = getattr(self, 'lambda_gamma', 1.0)
        l_d = getattr(self, 'lambda_delta', 1.0)
        l_k = getattr(self, 'lambda_block', 1.0)

        penalty += beta * l_b * (curr_f / F_REF) * SCALE  # fatigue
        penalty -= gamma * l_g * (wait_time / D_REF) * SCALE  # wait reward
        penalty -= delta * l_d * (e_idle * wait_h / E_REF) * SCALE  # emission reward

        my_dep = t + (v['stay'] * 2)
        for i, other in enumerate(self.berths[p_id]):
            if other:
                if (l_idx < i and my_dep < other['dep_t']) or (l_idx > i and my_dep > other['dep_t']):
                    penalty += l_k * SCALE  # blocking penalty
        return penalty

    def step(self, max_h=2000):
        t = self.t
        if t >= max_h * 2:
            return

        w_lvl = self.update_weather(t)
        is_lunch = (t % 48) in [24, 25]
        avail_tugs = 0 if (is_lunch or w_lvl == 3) else sum(1 for f in self.tug_free_time if f <= t)

        out_list = []
        for p, layers in self.berths.items():
            for i, v in enumerate(layers):
                if v and v['act_dep'] <= t: out_list.append((p, i, v))
        for p, i, v in sorted(out_list, key=lambda x: VESSEL_SPECS[x[2]['type']]['readiness'], reverse=True):
            spec = VESSEL_SPECS[v['type']]
            if w_lvl <= spec['weather_limit'] and avail_tugs >= spec['tugs']:
                #if self.mode == 'GA' and self.is_night(t):
                    # Multi-criteria night suppression for departures:
                    # Defer only if fatigue cost exceeds accumulated overdue delay
                  #  _beta = getattr(self, 'beta', 0.25)
                  #  _gamma = getattr(self, 'gamma', 0.25)
                 #   _delta = getattr(self, 'delta', 0.25)
                  #  _overdue = max(0, t - v.get('dep_t', t))  # ticks past planned departure
                  #  _f_cost = _beta * (self.calculate_vessel_fatigue(v['id'], v['type'], t) / F_REF)
                  #  _d_cost = _gamma * (_overdue / D_REF)
                  #  _e_cost = _delta * (VESSEL_SPECS[v['type']]['e_idle'] * _overdue * 0.5 / E_REF)
                  #  if _f_cost > (_d_cost + _e_cost):
                 #       v['act_dep'] += 1
                  #      self.delay += 0.5
                  #      continue
                blockers = [self.berths[p][j] for j in range(i + 1, len(self.berths[p])) if self.berths[p][j]]
                if not blockers:
                    assigned = 0
                    for tid in range(6):
                        if self.tug_free_time[tid] <= t and assigned < spec['tugs']:
                            self.tug_free_time[tid] = t + spec['duration']
                            assigned += 1
                    self.fatigue += self.calculate_vessel_fatigue(v['id'], v['type'], t)
                    self.counts["Total_Departure"] += 1
                    if self.record_log:
                        self.vessel_history.append(
                            {'Time': t / 2, 'VesselID': v['id'], 'Event': 'Departure', 'Loc': f"{p}-{i}",
                             'Weather': w_lvl, 'Tugs': avail_tugs})
                    self.berths[p][i] = None
                    avail_tugs -= spec['tugs']
                else:
                    self.shifting += 1
                    v['act_dep'] += 1
                    self.delay += 0.5
                    self.emissions += VESSEL_SPECS[v['type']]['e_underway'] * VESSEL_SPECS[v['type']]['duration']
            else:
                v['act_dep'] += 1
                self.delay += 0.5

        # Sort waiting vessels by mode-specific priority
        delta = getattr(self, 'delta', 0.25)

        def vessel_priority(v):
            if v['arr'] != t:
                return 0
            wt = max(0, t - v['arr_orig'])
            if self.mode == 'EDD':    return -(v.get('arr', 0) + v['stay'] * 2)  # earliest due first
            if self.mode == 'SPT':    return -v['stay']  # shortest stay first
            if self.mode == 'URGENT': return VESSEL_SPECS[v['type']]['readiness']  # highest readiness first
            # GA: prioritise by accumulated emissions
            return delta * VESSEL_SPECS[v['type']]['e_idle'] * wt * 0.5

        waiting_vessels = sorted(
            [v for v in self.scenario if v['arr'] == t],
            key=vessel_priority,
            reverse=True
        )
        all_vessels = waiting_vessels + [v for v in self.scenario if v['arr'] != t]

        for v in all_vessels:
            wait_time = max(0, t - v['arr_orig'])
            if v['arr'] == t:
                spec = VESSEL_SPECS[v['type']]
                if avail_tugs >= spec['tugs'] and w_lvl <= spec['weather_limit']:
                    if self.mode == 'GA' and self.is_night(t):
                        # Multi-criteria night suppression for arrivals:
                        # Compare fatigue cost of berthing NOW against ACCUMULATED delay+emission
                        # cost already incurred by waiting. Once accumulated cost exceeds fatigue
                        # cost, berth immediately regardless of time of day.
                        # Hard cap: never defer more than MAX_NIGHT_DEFER ticks (~12h)
                        MAX_NIGHT_DEFER = 6  # ticks = 3 hours (operational cap: no vessel held >6h for night timing)
                        _beta = getattr(self, 'beta', 0.25)
                        _gamma = getattr(self, 'gamma', 0.25)
                        _delta = getattr(self, 'delta', 0.25)
                        _e_idle = VESSEL_SPECS[v['type']]['e_idle']
                        # Cost of berthing now: one night fatigue event
                        _f_cost = _beta * (self.calculate_vessel_fatigue(v['id'], v['type'], t) / F_REF)
                        # Accumulated cost of waiting so far
                        _accum_d = _gamma * (wait_time / D_REF)
                        _accum_e = _delta * (_e_idle * wait_time * 0.5 / E_REF)
                        _accum_cost = _accum_d + _accum_e
                        # Defer only if fatigue cost exceeds accumulated cost AND under cap
                        if _f_cost > _accum_cost and wait_time < MAX_NIGHT_DEFER:
                            v['arr'] += 1
                            self.delay += 0.5
                            self.emissions += _e_idle * 0.5
                            continue
                        # else: berth now (either waited long enough or hit the cap)
                    best_s, best_p, best_l = 1e15, None, None
                    for p_id in PIER_CONFIG:
                        for l_idx in range(PIER_CONFIG[p_id]['layers']):
                            s = self.evaluate_fitness(v, p_id, l_idx, t, wait_time)
                            if s < best_s: best_s, best_p, best_l = s, p_id, l_idx
                    if best_p and best_s < 1e15:
                        assigned = 0
                        for tid in range(6):
                            if self.tug_free_time[tid] <= t and assigned < spec['tugs']:
                                self.tug_free_time[tid] = t + spec['duration']
                                assigned += 1
                        v['act_dep'] = t + (v['stay'] * 2)
                        v['dep_t'] = v['act_dep']
                        self.berths[best_p][best_l] = v
                        self.fatigue += self.calculate_vessel_fatigue(v['id'], v['type'], t)
                        self.counts["Total_Arrival"] += 1
                        if self.record_log:
                            self.vessel_history.append({
                                'Time': t / 2, 'VesselID': v['id'], 'Event': 'Arrival',
                                'Loc': f"{best_p}-{best_l}", 'Weather': w_lvl, 'Tugs': avail_tugs
                            })
                    else:
                        v['arr'] += 1
                        self.delay += 0.5
                        self.emissions += VESSEL_SPECS[v['type']]['e_idle'] * 0.5
                else:
                    v['arr'] += 1
                    self.delay += 0.5
                    self.emissions += VESSEL_SPECS[v['type']]['e_idle'] * 0.5

        self.t += 1

    def is_finished(self, max_h=2000):
        return self.t >= max_h * 2

    def run(self, max_h=2000):
        self.reset()
        for _ in range(max_h * 2):
            self.step(max_h=max_h)
        return {
            'shifting': self.shifting,
            'fatigue': self.fatigue,
            'delay': self.delay,
            'emissions': round(self.emissions, 1),  # kg CO2
            'counts': dict(self.counts),
            'history': self.vessel_history
        }


def generate_scenario(max_h=2000):
    scen = []
    for tc, info in VESSEL_SPECS.items():
        for i in range(info['count']):
            curr_h = random.randint(0, info['cycle'])
            while curr_h < max_h:
                scen.append({'id': f'{tc}{i}_{curr_h}', 'type': tc, 'arr': curr_h * 2, 'arr_orig': curr_h * 2,
                             'stay': random.randint(*info['stay_range'])})
                curr_h += info['cycle']
    return sorted(scen, key=lambda x: x['arr'])























































