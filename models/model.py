# ================================================
# 📂 trading_app/model.py
# ================================================
from stable_baselines3 import PPO

def train_model(env, config):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=config["total_timesteps"])
    model.save(config["model_save_path"])
    return model

