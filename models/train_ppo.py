"""
train.py
---------
Training script for RL-based trading agent using PPO with a realistic market simulation.
Includes:
 - Configurable training/testing split
 - TensorBoard logging
 - Evaluation callback with early stopping
 - Modular design for easy extensions
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from envs.config import CONFIG



# ========================
# MAIN TRAINING LOGIC
# ========================
def train_ppo_model(env, eval_env):
    # 4) PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=CONFIG["tensorboard_log_dir"],
        ent_coef=0.005,              # entropy regularization for exploration
        learning_rate=3e-4,
        gamma=0.99,                  # discount factor for long-term reward
        gae_lambda=0.95,             # GAE smoothing
        batch_size=256
    )

    # 5) Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='../' + CONFIG["model_save_path"],
        log_path=CONFIG["tensorboard_log_dir"],
        eval_freq=5000,
        deterministic=True,
        render=False,
        callback_on_new_best=StopTrainingOnRewardThreshold(reward_threshold=1_000, verbose=1)
    )

    # 6) Train
    model.learn(total_timesteps=CONFIG["total_timesteps"], callback=eval_callback)

    # 7) Save
    model.save(os.path.join('../' + CONFIG["model_save_path"], "final_model"))

    print("✅ Training complete. Model saved at:", CONFIG["model_save_path"])
