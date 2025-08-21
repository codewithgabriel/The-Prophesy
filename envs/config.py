
# ========================
# CONFIGURATION
# ========================
CONFIG = {
    # Meta data
    "asset_symbol": "AAPL",
    "csv_path": "dataset/aapl.csv",  # if None, fetch from API
    "start_date": "2015-01-01",
    "end_date": "2023-01-01",
    "train_split": 0.8,
    
    # Environment core
    "window_size": 50,
    "initial_balance": 100.0,
    
    # Costs & execution
    "commission_pct": 0.001,           # replaces transaction_cost_pct
    "commission_fixed": 0.0,           # new
    "spread_pct": 0.0001,
    "slippage_coeff": 0.0002,          # replaces slippage_pct
    "volume_limit": 0.1,               # max fraction of observed volume we can consume per step

    # scaling
    "max_position_size": 10_000,  # cap at 10k shares
    "max_risk_per_trade": 0.02,
    "stop_loss_pct": 0.02,
    "drawdown_scale_threshold": 0.1,
    "drawdown_scale_factor": 0.5,
    "volatility_scaling": True,
    
    # Leverage & margin
    "max_leverage": 2.0,
    "maintenance_margin": 0.25,        # new
    
    # Financing & rewards
    "financing_rate_annual": 0.02,     # new
    "reward_scaling": 1.0,             # new
    "dd_penalty_coeff": 0.0,           # new
    "turnover_penalty_coeff": 0.0,     # new
    
    # Observations & episodes
    "normalize_observations": True,    # new
    "random_start": True,              # new
    "episode_length": None,            # new
    
    # Training
    "total_timesteps": 300_000,
    
    # Logging & saving
    "tensorboard_log_dir": "saved_models/ppo/tb_logs",
    "model_save_path": "saved_models/ppo_trader_model/final_model.zip"
}
