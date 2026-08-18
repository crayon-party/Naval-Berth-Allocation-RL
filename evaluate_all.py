"""
evaluate_all.py
===============
Final benchmark for the paper: MILP (exact reference) vs Greedy constructive
multi-objective scheduler vs FCFS vs Random vs trained DRL policy, on
IDENTICAL held-out instances.

Requirements:
  * naval_berth_env.py in the same folder
  * your MILP benchmark script saved as  naval_milp_benchmark.py  (same folder)
    -> needs `build_milp(scenario, horizon_h, time_limit_s)` importable
  * ppo_berth_best.zip from train_ppo.py (omit --model to skip DRL)

Run:
    python evaluate_all.py --seeds 100 120 --horizon 240 --model ppo_berth_best
Outputs:
    benchmark_results.csv  +  console summary table (paste into paper)

Objective used for gaps: shifting + fatigue_arr + delay_h  (MILP-comparable:
the MILP counts fatigue at the berthing event only). The full operational
objective (incl. departure fatigue) is also recorded per method.
"""

import argparse
import time
import numpy as np
import pandas as pd

from naval_berth_env import (NavalBerthEnv, GreedyPolicy, FCFSPolicy,
                             RandomPolicy, generate_scenario)

try:
    from naval_milp_benchmark import build_milp
    HAVE_MILP = True
except ImportError:
    HAVE_MILP = False
    print("[warn] naval_milp_benchmark.py not importable -> skipping MILP column")


def run_policy(policy, horizon, seed, weather, queue_rule="wait"):
    env = NavalBerthEnv(horizon_h=horizon, scenario_seed=seed,
                        weather_prob=weather, lunch_break=False,
                        queue_rule=queue_rule)
    obs, info = env.reset(seed=seed)
    done = info.get('n_vessels', 0) == 0
    t0 = time.perf_counter()
    n_dec = 0
    final = env.metrics() if done else None
    while not done:
        a = policy(env, obs)
        n_dec += 1
        obs, r, done, _, info = env.step(a)
        if done:
            final = info
    final['decision_time_ms'] = ((time.perf_counter() - t0) / max(n_dec, 1)) * 1e3
    return final


def wrap_simple(pol):
    return lambda env, obs: pol(env)


def wrap_model(model):
    def f(env, obs):
        a, _ = model.predict(obs, action_masks=env.action_masks(),
                             deterministic=True)
        return int(a)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs=2, default=[100, 120],
                    help='seed range [lo, hi)')
    ap.add_argument('--horizon', type=int, default=240)
    ap.add_argument('--weather', type=float, default=0.0)
    ap.add_argument('--model', type=str, default=None)
    ap.add_argument('--milp_timelimit', type=int, default=600)
    args = ap.parse_args()

    policies = {'FCFS': (wrap_simple(FCFSPolicy()), 'wait'),
                'EDD': (wrap_simple(FCFSPolicy()), 'edd'),
                'SPT': (wrap_simple(FCFSPolicy()), 'spt'),
                'URGENT': (wrap_simple(FCFSPolicy()), 'urgent'),
                'Random': (wrap_simple(RandomPolicy(0)), 'wait'),
                'Greedy': (wrap_simple(GreedyPolicy()), 'wait')}
    if args.model:
        from sb3_contrib import MaskablePPO
        policies['DRL'] = (wrap_model(MaskablePPO.load(args.model)), 'wait')

    rows = []
    for seed in range(args.seeds[0], args.seeds[1]):
        scen_h = generate_scenario(args.horizon, seed=seed)
        if not scen_h:
            print(f"seed {seed}: empty scenario, skipped")
            continue
        row = {'seed': seed, 'n_vessels': len(scen_h)}

        if HAVE_MILP:
            m = build_milp(scen_h, horizon_h=args.horizon,
                           time_limit_s=args.milp_timelimit, verbose=False)
            row['MILP_obj'] = m['obj_value']
            row['MILP_status'] = m['status']
            row['MILP_time_s'] = m['solve_time_s']
            row['MILP_shift'] = m['shifting']
            row['MILP_fatigue'] = m['fatigue']
            row['MILP_delay'] = m['delay']

        for name, (pol, qrule) in policies.items():
            res = run_policy(pol, args.horizon, seed, args.weather, queue_rule=qrule)
            row[f'{name}_obj'] = res['obj_full']
            row[f'{name}_full'] = res['obj_full']
            row[f'{name}_shift'] = res['shifting']
            row[f'{name}_delay'] = res['delay_h']
            row[f'{name}_unserved'] = res['unserved']
            row[f'{name}_ms_per_decision'] = round(res['decision_time_ms'], 3)

        if HAVE_MILP and row.get('MILP_obj'):
            for name in policies:
                row[f'{name}_gap_pct'] = round(
                    (row[f'{name}_obj'] - row['MILP_obj'])
                    / row['MILP_obj'] * 100, 1)
        rows.append(row)
        print(f"seed {seed}: " + "  ".join(
            f"{k}={row.get(k)}" for k in
            (['MILP_obj'] if HAVE_MILP else []) +
            [f'{n}_obj' for n in policies]))

    df = pd.DataFrame(rows)
    df.to_csv("benchmark_results.csv", index=False)

    print("\n" + "=" * 70)
    print(f"SUMMARY over {len(df)} instances "
          f"(horizon={args.horizon}h, weather={args.weather})")
    print("=" * 70)
    if HAVE_MILP and 'MILP_obj' in df:
        print(f"{'MILP':<8} mean obj {df['MILP_obj'].mean():9.1f}   "
              f"mean solve {df['MILP_time_s'].mean():7.1f} s")
    for name in policies:
        line = (f"{name:<8} mean obj {df[f'{name}_obj'].mean():9.1f}   "
                f"mean ms/decision {df[f'{name}_ms_per_decision'].mean():7.3f}")
        if f'{name}_gap_pct' in df:
            line += f"   mean gap vs MILP {df[f'{name}_gap_pct'].mean():+6.1f}%"
        u = df[f'{name}_unserved'].sum()
        if u:
            line += f"   [!] unserved total: {u}"
        print(line)
    print("\nSaved -> benchmark_results.csv")


if __name__ == "__main__":
    main()
