# Naval-Berth-Allocation-RL

Code accompanying **"Learning When to Wait: Action-Masked Reinforcement Learning
for Real-Time Naval Berth Allocation"** (MICAI 2026).

This repository contains the simulation environment, RL training pipeline, and
MILP benchmark used to produce the results reported in the paper (Table 5 and
associated figures).

## Components

- `naval_berth_env.py` — Gym-compatible simulation environment (`NavalBerthEnv`),
  implementing the MDP formulation of Section 5.3 (feasibility-based action
  masking, bounded deferral action, 89-dimensional observation space). Also
  includes the CMOS/`GreedyPolicy` baseline.
- `naval_milp_benchmark.py` — offline MILP lower bound (PuLP + CBC),
  producing the solver-certified dual bounds reported in Section 5.1 and Table 5.
- `train_ppo.py` — Maskable PPO training pipeline (sb3-contrib /
  Stable-Baselines3), hyperparameters as in Table 4.
- `evaluate_all.py` — evaluation harness that runs all baselines
  (FCFS, EDD, SPT, URGENT, masked random, CMOS, RL) on the held-out test
  set and reproduces Table 5.
- `ppo_best_s0.zip`–`ppo_best_s7.zip` — best-validation checkpoint per seed
  (8 seeds, matching Section 5.3's protocol). **These are what Table 5's
  reported results are built from.**
- `ppo_final_s0.zip`–`ppo_final_s7.zip` — end-of-training checkpoint per
  seed, included for transparency into the full run. Not used for any
  reported number.
- `results/benchmark_results_s0.csv`–`s7.csv` — per-instance evaluation
  output for all 8 seeds against the 25-instance test set (seeds 130–154).

## Requirements

See `requirements.txt`. Developed and tested with:

| Component | Version |
|---|---|
| Python | 3.10.19 |
| gymnasium | 0.29.1 |
| stable-baselines3 | 2.3.2 |
| sb3-contrib | 2.3.0 |
| torch | 2.2.2 |
| PuLP | 3.3.0 |
| CBC | 2.10.3 |

Install with:

```bash
pip install -r requirements.txt
```

CBC must be available on your system `PATH` (see [CBC installation
instructions](https://github.com/coin-or/Cbc)) for the MILP benchmark to run.

## Reproducing paper results

1. Run `naval_milp_benchmark.py` to generate the offline lower bounds for the
   25 test instances (seeds 130–154).
2. Run `train_ppo.py` to train the Maskable PPO policy (8 seeds, ~10 minutes
   per seed on a desktop CPU).
3. Run `evaluate_all.py` to evaluate all baselines and the trained policy on
   the held-out test set, producing the numbers in Table 5. Use `--skip_milp`
   after the first seed's evaluation to avoid redundantly re-solving the MILP
   bound, since it is identical across all seeds.

**Note on reproducibility:** checkpoints in this repository were retrained
independently of the results reported in the paper. Individual training runs
may vary due to the stochastic nature of PPO, but results consistently fall
within the paper's reported range across seeds (504.7–688.0, Section 6.1),
and every independently retrained seed outperforms the CMOS baseline (778.2),
consistent with the paper's claims.

## Digital twin / Unity frontend

## Digital twin / Unity frontend

The Unity 3D operator interface and FastAPI server described in Section 3
are visualization and interaction layers; they do not affect any reported
quantitative result and are not required to reproduce the paper's
experiments (`naval_berth_env.py`, `naval_milp_benchmark.py`, `train_ppo.py`,
and `evaluate_all.py` are fully sufficient for that).
 
The Unity project itself is not included in this repository. It depends on
third-party Unity Asset Store content (3D vessel and shipyard models) whose
standard license does not permit redistribution outside the Asset Store. 

The Unity project source, excluding those licensed assets, is available from the
corresponding author upon request.

In place of the Unity project, this repository includes:
 
- [`docs/unity_operator_interface.png`](docs/unity_operator_interface.png) —
  an annotated overview of the operator interface, showing how each panel
  maps onto the monitoring / prediction / decision-making structure
  described in Section 3.
- [`docs/demo-v2_highres.mp4`](docs/demo-v2_highres.mp4) — a recorded demonstration of the digital
  twin in operation, showing scenario setup, weather-event injection, and
  live rescheduling.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{singh2026berth,
  title     = {Learning When to Wait: Action-Masked Reinforcement Learning for Real-Time Naval Berth Allocation},
  author    = {Singh, Kanika and Kim, Heesun and Baek, Jiwon and Woo, Jong Hun},
  booktitle = {MICAI 2026},
  year      = {2026}
}
```

## License

[Add license here]