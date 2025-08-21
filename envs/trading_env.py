import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class TradingEnv(gym.Env):
    """A Gymnasium environment that models many practical aspects of trading.

    Observation: a 1D numpy array combining a flattened window of features and a small
    portfolio vector. All inputs are scaled to reasonable ranges (approx -10..10)
    for numerical stability.

    Action: continuous Box(shape=(1,), low=-1.0, high=1.0) representing target exposure
    as fraction of net worth. Example: action=0.25 => target long exposure 25% of net worth.
    Negative actions indicate short exposure. Leverage is limited by max_leverage.

    Reward: log return of net_worth between steps, optionally penalized for drawdown and turnover.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols=None,
        window_size: int = 50,
        initial_balance: float = 100_000.0,
        commission_pct: float = 0.0005,  # 5 bps
        commission_fixed: float = 0.0,
        spread_pct: float = 0.0002,  # 2 bps round-trip (half-spread = 1 bp)
        slippage_coeff: float = 0.1,  # coef controlling slippage vs trade/volume
        volume_limit: float = 0.1,  # max fraction of observed volume we can consume per step.
        max_leverage: float = 3.0,  # max gross exposure / net_worth (abs exposure cap).
        maintenance_margin: float = 0.25,  # fraction of net_worth as maintenance margin.
        financing_rate_annual: float = 0.02,  # annual interest on borrowed capital.
        reward_scaling: float = 1.0,  # Scalar multiplier on the base reward.
        dd_penalty_coeff: float = 0.0,  # Penalize drawdown proportionally if > 0.
        turnover_penalty_coeff: float = 0.0,  # Penalize trading volume via last-trade notional.
        normalize_observations: bool = True,  # Whether to z-score features.
        random_start: bool = True,  # Randomize start index each episode.
        episode_length: int | None = None,  # If set, hard cap steps per episode.
    ):
        super().__init__()

        # Basic data checks
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        self.raw_df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols if feature_cols is not None else ["Close"]
        self.window_size = int(window_size)

        # Economic params
        self.initial_balance = float(initial_balance)
        self.commission_pct = float(commission_pct)
        self.commission_fixed = float(commission_fixed)
        self.spread_pct = float(spread_pct)
        self.slippage_coeff = float(slippage_coeff)
        self.volume_limit = float(volume_limit)
        self.max_leverage = float(max_leverage)
        self.maintenance_margin = float(maintenance_margin)
        self.financing_rate_annual = float(financing_rate_annual)
        self.reward_scaling = float(reward_scaling)
        self.dd_penalty_coeff = float(dd_penalty_coeff)
        self.turnover_penalty_coeff = float(turnover_penalty_coeff)

        # Episode mechanics
        self.random_start = bool(random_start)
        self.episode_length = episode_length

        # Precompute a few helpful series
        self.raw_df["avg_volume"] = self.raw_df["Volume"].rolling(window=20, min_periods=1).mean()

        # Normalization helpers
        self.normalize_observations = bool(normalize_observations)
        if self.normalize_observations:
            self._feature_means = self.raw_df[self.feature_cols].mean()
            self._feature_stds = self.raw_df[self.feature_cols].std().replace(0, 1.0)
        else:
            self._feature_means = pd.Series(0.0, index=self.feature_cols)
            self._feature_stds = pd.Series(1.0, index=self.feature_cols)

        # Observation space
        self.n_features = len(self.feature_cols)
        self.obs_window_shape = (self.window_size, self.n_features)
        flat_len = self.window_size * self.n_features + 3
        obs_low = -10.0 * np.ones(flat_len, dtype=np.float32)
        obs_high = 10.0 * np.ones(flat_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state
        self._reset_internal_state()

    # ----------------------- internal state & helpers -----------------------
    def _reset_internal_state(self):
        self.step_count = 0
        self.position_shares = 0.0
        self.entry_value = 0.0
        self.balance = float(self.initial_balance)
        self.net_worth = float(self.initial_balance)
        self.trades = []
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.start_index = self.window_size
        self.current_index = self.start_index
        self.terminated = False
        self.cum_fees = 0.0
        self.cum_financing = 0.0
        self.cum_turnover = 0.0  # Track cumulative turnover for better penalty calculation

    def seed(self, seed=None):
        np.random.seed(seed)

    # ----------------------- data helpers -----------------------
    def _set_start_index(self):
        max_start = len(self.raw_df) - 1
        if self.episode_length is not None:
            max_start = max_start - self.episode_length
        min_start = self.window_size
        if min_start >= len(self.raw_df):
            raise ValueError("Data too short for chosen window_size")
        if self.random_start:
            self.start_index = np.random.randint(min_start, max_start + 1)
        else:
            self.start_index = min_start
        self.current_index = int(self.start_index)

    def _get_price_at(self, idx: int) -> float:
        return float(self.raw_df.loc[idx, "Close"])

    def _get_avg_volume_at(self, idx: int) -> float:
        return float(self.raw_df.loc[idx, "avg_volume"])

    # ----------------------- market microstructure models -----------------------
    def _estimate_exec_price(self, side: str, price: float, trade_value: float, avg_volume: float) -> float:
        half_spread = 0.5 * self.spread_pct * price
        if side == "buy":
            price_after_spread = price + half_spread
        else:
            price_after_spread = price - half_spread

        trade_shares_est = max(0.0, trade_value / (price + 1e-12))
        max_fillable = max(1.0, avg_volume * self.volume_limit)
        volume_ratio = min(1.0, trade_shares_est / max_fillable)
        slippage = self.slippage_coeff * (volume_ratio ** 1.2) * price

        if side == "buy":
            exec_price = price_after_spread * (1.0 + slippage / (price + 1e-12))
        else:
            exec_price = price_after_spread * (1.0 - slippage / (price + 1e-12))

        return float(exec_price)

    # ----------------------- core trade execution -----------------------
    def _execute_target_exposure(self, target_frac: float, price: float, avg_volume: float):
        # Cap target_frac by max_leverage
        target_frac = float(np.clip(target_frac, -self.max_leverage, self.max_leverage))

        desired_position_value = target_frac * self.net_worth
        current_position_value = self.position_shares * price
        delta_value = desired_position_value - current_position_value

        if abs(delta_value) < 1e-8:
            return 0.0, 0.0

        side = "buy" if delta_value > 0 else "sell"
        intended_trade_value = abs(delta_value)

        intended_shares = intended_trade_value / (price + 1e-12)
        max_fillable_shares = max(1.0, avg_volume * self.volume_limit)
        executed_shares = float(min(intended_shares, max_fillable_shares))
        executed_value = executed_shares * price

        if executed_shares <= 0:
            return 0.0, 0.0

        exec_price = self._estimate_exec_price(side=side, price=price, trade_value=executed_value, avg_volume=avg_volume)
        self.exec_price = exec_price
        trade_notional = executed_shares * exec_price
        commission = abs(trade_notional) * self.commission_pct + self.commission_fixed
        self.cum_fees += commission

        signed_shares = executed_shares if delta_value > 0 else -executed_shares
        self.position_shares += signed_shares
        
        # FIXED: Commission always reduces cash regardless of trade direction
        cash_flow = -signed_shares * exec_price - commission
        self.balance += cash_flow

        self.entry_value = self.position_shares * exec_price

        return float(np.sign(delta_value) * executed_value), float(commission)

    # ----------------------- observation & reward -----------------------
    def _get_obs(self):
        start = int(self.current_index - self.window_size)
        end = int(self.current_index - 1)
        window = self.raw_df.loc[start : end, self.feature_cols].values.copy()

        if self.normalize_observations:
            window = (window - self._feature_means.values) / (self._feature_stds.values + 1e-12)

        flat_window = window.flatten()

        pv = self.position_shares * self._get_price_at(self.current_index)
        cash_frac = 0.0 if self.net_worth == 0 else self.balance / (self.net_worth + 1e-12)
        pos_frac = 0.0 if self.net_worth == 0 else pv / (self.net_worth + 1e-12)
        pos_shares_normed = self.position_shares / (self._get_avg_volume_at(self.current_index) + 1e-12)
        portfolio_vec = np.array([cash_frac, pos_frac, pos_shares_normed], dtype=np.float32)

        obs = np.concatenate([flat_window.astype(np.float32), portfolio_vec.astype(np.float32)])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _compute_step_reward(self):
        if self.prev_net_worth <= 0:
            raw = 0.0
        else:
            raw = np.log((self.net_worth + 1e-12) / (self.prev_net_worth + 1e-12))
        
        dd = 0.0
        if self.net_worth < self.max_net_worth:
            dd = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-12)
        dd_pen = self.dd_penalty_coeff * dd
        
        # Improved turnover penalty: use absolute change in position value
        current_pos_value = self.position_shares * self._get_price_at(self.current_index)
        prev_pos_value = (self.position_shares - self.trades[-1]["position_shares"] if self.trades 
                         else 0) * self._get_price_at(self.current_index - 1) if self.current_index > 0 else 0
        turnover = abs(current_pos_value - prev_pos_value) if self.trades else 0.0
        turnover_pen = self.turnover_penalty_coeff * turnover / (self.initial_balance + 1e-12)

        reward = self.reward_scaling * raw - dd_pen - turnover_pen
        return float(reward)

    # ----------------------- public gym methods -----------------------
    def reset(self, seed=None, options=None):
        self.seed(seed)
        self._reset_internal_state()
        self._set_start_index()
        self.net_worth = float(self.initial_balance)
        self.prev_net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self.current_index = int(self.start_index)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} invalid"
        target_frac = float(action[0])

        price = self._get_price_at(self.current_index)
        avg_volume = self._get_avg_volume_at(self.current_index)

        executed_notional, commission = self._execute_target_exposure(target_frac, price, avg_volume)

        
        profit = (price - self.exec_price) * self.position_shares
        
        if abs(executed_notional) > 0:
            self.trades.append({
                "index": int(self.current_index),
                "notional": float(executed_notional),
                "commission": float(commission),
                "position_shares": float(self.position_shares),
                "profit": float(profit),
            })

        position_value = self.position_shares * price
        borrowed = max(0.0, abs(position_value) - self.net_worth)
        daily_rate = self.financing_rate_annual / 252.0
        financing_cost = borrowed * daily_rate
        self.balance -= financing_cost
        self.cum_financing += financing_cost

        self.net_worth = self.balance + position_value
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        margin_requirement = self.maintenance_margin * abs(position_value)
        if self.net_worth <= 0 or (abs(position_value) > 0 and self.net_worth < margin_requirement):
            if self.position_shares != 0:
                # FIXED: Correct liquidation cash flow calculation
                close_shares = -self.position_shares
                close_value = abs(close_shares) * price
                commission_close = close_value * self.commission_pct
                
                # Correct cash flow calculation for liquidation
                cash_flow = -close_shares * price
                self.balance += cash_flow - commission_close
                
                self.cum_fees += commission_close
                self.trades.append({
                    "index": int(self.current_index),
                    "notional": float(-np.sign(self.position_shares) * close_value),
                    "commission": float(commission_close),
                    "position_shares": 0.0,
                    "liquidation": True,
                })
                self.position_shares = 0.0
                position_value = 0.0
            
            self.net_worth = self.balance + position_value
            info = {"reason": "margin_call_or_bankrupt"}
            reward = self._compute_step_reward()
            self.prev_net_worth = self.net_worth
            self.current_index += 1
            self.terminated = True
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, float(reward), True, False, info

        reward = self._compute_step_reward()

        self.prev_net_worth = self.net_worth
        self.current_index += 1
        self.step_count += 1

        done = False
        if self.episode_length is not None and (self.current_index - self.start_index) >= self.episode_length:
            done = True
        if self.current_index >= len(self.raw_df) - 1:
            done = True

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)

        info = {
            "net_worth": float(self.net_worth),
            "cum_fees": float(self.cum_fees),
            "cum_financing": float(self.cum_financing),
            "position_shares": float(self.position_shares),
            "start_index": int(self.start_index),
        }

        return obs, float(reward), bool(done), False, info

    def render(self, mode="human"):
        price = self._get_price_at(self.current_index)
        print(
            f"Idx:{self.current_index} Price:{price:.4f} NW:{self.net_worth:.2f} Bal:{self.balance:.2f} PosShares:{self.position_shares:.4f}"
        )