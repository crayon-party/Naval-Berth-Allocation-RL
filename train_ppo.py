"""
train_ppo.py
============
Train a MaskablePPO policy on NavalBerthEnv.

Setup (Mac mini or Colab):
    pip install gymnasium "stable-baselines3>=2.3" sb3-contrib torch tensorboard

Run:
    python train_ppo.py --steps 500000 --horizon 240
    tensorboard --logdir runs/        # watch ep_rew_mean — THE GATE METRIC

Gate check (tomorrow evening): ep_rew_mean must be clearly rising and the
periodic eval printout must beat Random and approach/beat Greedy on the
held-out seeds. If it is flat after ~200k steps, switch to the MILP paper.
"""

import argparse
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.monitor import Monitor

from naval_berth_env import (NavalBerthEnv, GreedyPolicy, FCFSPolicy,
                             RandomPolicy, rollout)


def mask_fn(env):
    return env.action_masks()


EVAL_SEEDS = list(range(100, 110))   # held-out; training uses random seeds


def make_env(horizon, weather_prob, rank):
    def _f():
        env = NavalBerthEnv(horizon_h=horizon, scenario_seed=None,
                            weather_prob=weather_prob, lunch_break=False)
        env.reset(seed=10_000 + rank)
        return ActionMasker(Monitor(env), mask_fn)
    return _f


def evaluate_policy_fn(model, horizon, weather_prob, seeds=EVAL_SEEDS):
    objs = []
    for s in seeds:
        env = NavalBerthEnv(horizon_h=horizon, scenario_seed=s,
                            weather_prob=weather_prob, lunch_break=False)
        obs, info = env.reset(seed=s)
        done = info.get('n_vessels', 0) == 0
        final = env.metrics() if done else None
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, done, _, info = env.step(int(action))
            if done:
                final = info
        objs.append(final['obj_milp_comparable'] + 1000 * final['unserved'])
    return float(np.mean(objs))


class EvalGate(BaseCallback):
    def __init__(self, horizon, weather_prob, every=25_000):
        super().__init__()
        self.horizon, self.wp, self.every = horizon, weather_prob, every
        self.best = float('inf')

    def _on_step(self):
        if self.num_timesteps % self.every < self.training_env.num_envs:
            score = evaluate_policy_fn(self.model, self.horizon, self.wp)
            self.logger.record("eval/heldout_obj", score)
            print(f"  [eval @ {self.num_timesteps:>8}] mean obj on held-out: {score:.1f} "
                  f"(best {self.best:.1f})")
            if score < self.best:
                self.best = score
                self.model.save("ppo_berth_best")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=500_000)
    ap.add_argument('--horizon', type=int, default=240)
    ap.add_argument('--weather', type=float, default=0.0)
    ap.add_argument('--nenv', type=int, default=8)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent", type=float, default=0.03)
    ap.add_argument("--net", type=int, default=256)
    args = ap.parse_args()

    # Baseline reference on held-out seeds (what DRL must approach)
    print("Baselines on held-out seeds (obj = shifting + fatigue_arr + delay_h):")
    for name, pol in (('FCFS', FCFSPolicy()), ('Greedy', GreedyPolicy()),
                      ('Random', RandomPolicy(0))):
        vals = []
        for s in EVAL_SEEDS:
            env = NavalBerthEnv(horizon_h=args.horizon, scenario_seed=s,
                                weather_prob=args.weather, lunch_break=False)
            m = rollout(env, pol, seed=s)
            vals.append(m['obj_milp_comparable'] + 1000 * m['unserved'])
        print(f"  {name:<8} mean obj = {np.mean(vals):8.1f}")

    # DummyVecEnv: in-process, avoids macOS spawn/pickle issues; env is light enough
    venv = DummyVecEnv([make_env(args.horizon, args.weather, i) for i in range(args.nenv)])

    model = MaskablePPO(
        "MlpPolicy", venv, verbose=1, seed=args.seed,
        n_steps=256, batch_size=512, n_epochs=10,
        learning_rate=args.lr, gamma=0.99, gae_lambda=0.95,
        ent_coef=args.ent, clip_range=0.2,
        policy_kwargs=dict(net_arch=[args.net, args.net]),
        tensorboard_log="runs/",
    )

    model.learn(total_timesteps=args.steps,
                callback=EvalGate(args.horizon, args.weather),
                progress_bar=True)
    model.save("ppo_berth_final")
    print("Saved: ppo_berth_final.zip  (best checkpoint: ppo_berth_best.zip)")


if __name__ == "__main__":
    main()