# Naval-Berth-Allocation-RL

Code accompanying **"Learning When to Wait: Action-Masked Reinforcement Learning
for Real-Time Naval Berth Allocation"** (MICAI 2026).

This repository contains the simulation environment, RL training pipeline, and
MILP benchmark used to produce the results reported in the paper (Table 5 and
associated figures).

> **Status:** actively being populated. This repository is being built out in
> two phases: (1) the simulation environment, RL training pipeline, and MILP
> benchmark used to produce the paper's reported results, followed by (2) the
> Unity 3D digital-twin operator interface and FastAPI server described in
> Section 3. Phase 2 depends on resolving third-party Unity Asset Store
> licensing before those files can be published.

## Components

- [ ] `src/naval_berth_env.py` — Gym-compatible simulation environment (`NavalBerthEnv`),
      implementing the MDP formulation of Section 5.3 (feasibility-based action
      masking, bounded deferral action, 89-dimensional observation space).
- [ ] `src/naval_milp_benchmark.py` — offline MILP lower bound (PuLP + CBC),
      producing the solver-certified dual bounds reported in Section 5.1 and Table 5.
- [ ] `src/train_ppo.py` — Maskable PPO training pipeline (sb3-contrib /
      Stable-Baselines3), hyperparameters as in Table 4.
- [ ] `src/evaluate_all.py` — evaluation harness that runs all baselines
      (FCFS, EDD, SPT, URGENT, masked random, CMOS, RL) on the held-out test
      set and reproduces Table 5.
- [ ] `configs/ppo_hyperparams.yaml` — training hyperparameters as a standalone
      config file.
- [ ] `results/results_clear.csv` — raw sweep results underlying the paper's
      reported metrics.
- [ ] `unity/` — Unity 3D digital-twin operator interface (Section 3). Pending
      third-party asset licensing review; not yet included.
- [ ] `server/` — FastAPI middleware connecting the simulation core to the
      Unity frontend (Section 3, Table 3). Not yet included.

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

Instructions will be added here as each component is uploaded. The intended
workflow:

1. Run `naval_milp_benchmark.py` to generate the offline lower bounds for the
   25 test instances (seeds 130–154).
2. Run `train_ppo.py` to train the Maskable PPO policy (8 seeds, ~10 minutes
   per seed on a desktop CPU).
3. Run `evaluate_all.py` to evaluate all baselines and the trained policy on
   the held-out test set, producing the numbers in Table 5.

## Digital twin / Unity frontend

The Unity 3D operator interface and FastAPI server described in Section 3
are visualization and interaction layers; they do not affect any reported
quantitative result and are not required to reproduce the paper's
experiments (`naval_berth_env.py`, `naval_milp_benchmark.py`, `train_ppo.py`,
and `evaluate_all.py` are fully sufficient for that). They will be added to
this repository once third-party Unity Asset Store licensing has been
reviewed.

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
