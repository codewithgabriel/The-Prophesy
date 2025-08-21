"""
realistic_trading_env.py

A more realistic single-instrument trading environment (Gymnasium-compatible)
Designed with pragmatic best-practices for research / paper-trading simulation.

Key features implemented:
- Continuous action space: target exposure fraction in [-1, 1] (negative => short)
- Position sizing w.r.t. net worth, supports leverage up to `max_leverage`
- Explicit commission (percentage + fixed) and bid-ask spread
- Simple slippage model proportional to trade size vs. recent volume
- Partial fills limited by available market volume per step
- Margin tracking and liquidation when maintenance margin breached
- Overnight financing (carry) for leveraged positions
- Observation includes normalized price/indicator window + normalized portfolio state
- Randomized episode start (walk-forward style) and per-step time limit
- Reward: log return of net worth per step with optional drawdown penalty and turnover penalty

Notes:
- This is still a simulator; for real deployment add micro-structure (order book, limit orders, fills based on price paths), realistic intraday volume curves, and market impact models.
- The environment expects `df` to contain at least the following columns: ['Open','High','Low','Close','Volume'] plus any feature columns you compute beforehand.

Author: ChatGPT (experienced ML/trading engineer)
"""

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
        volume_limit: float = 0.1,  # max fraction of observed volume we can consume per step
        max_leverage: float = 3.0,  # max gross exposure / net_worth
        maintenance_margin: float = 0.25,  # fraction of net_worth as maintenance margin
        financing_rate_annual: float = 0.02,  # annual interest on borrowed capital
        reward_scaling: float = 1.0,
        dd_penalty_coeff: float = 0.0,
        turnover_penalty_coeff: float = 0.0,
        normalize_observations: bool = True,
        random_start: bool = True,
        episode_length: int | None = None,
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
        self.episode_length = episode_length  # None => until end of df

        # Precompute a few helpful series
        # We will use a short moving average of volume as a proxy for available liquidity
        self.raw_df["avg_volume"] = self.raw_df["Volume"].rolling(window=20, min_periods=1).mean()

        # Normalization helpers: compute mean/std per feature over full dataset
        self.normalize_observations = bool(normalize_observations)
        if self.normalize_observations:
            self._feature_means = self.raw_df[self.feature_cols].mean()
            self._feature_stds = self.raw_df[self.feature_cols].std().replace(0, 1.0)
        else:
            self._feature_means = pd.Series(0.0, index=self.feature_cols)
            self._feature_stds = pd.Series(1.0, index=self.feature_cols)

        # Observation space: flattened window + portfolio vector
        self.n_features = len(self.feature_cols)
        self.obs_window_shape = (self.window_size, self.n_features)
        # we'll flatten the window; then append [cash/net_worth, position_value/net_worth, position_fraction]
        flat_len = self.window_size * self.n_features + 3
        # Bound observations to a conservative range for numerical stability
        obs_low = -10.0 * np.ones(flat_len, dtype=np.float32)
        obs_high = 10.0 * np.ones(flat_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: continuous target exposure fraction in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Internal state
        self._reset_internal_state()

    # ----------------------- internal state & helpers -----------------------
    def _reset_internal_state(self):
        """Reset all per-episode variables. Called by reset()."""
        self.step_count = 0
        # position expressed as signed number of shares (positive long, negative short)
        self.position_shares = 0.0
        self.entry_value = 0.0  # signed value of position at entry
        self.balance = float(self.initial_balance)  # cash position
        self.net_worth = float(self.initial_balance)
        self.gross_exposure = 0.0  # |position_value| / net_worth
        self.trades = []
        self.max_net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.start_index = self.window_size
        self.current_index = self.start_index
        self.terminated = False
        self.cum_fees = 0.0
        self.cum_financing = 0.0

    def seed(self, seed=None):
        np.random.seed(seed)

    # ----------------------- data helpers -----------------------
    def _set_start_index(self):
        """Randomize or set the start index for the episode.

        Ensures there is enough room for `window_size` history and `episode_length` future.
        """
        max_start = len(self.raw_df) - 1
        if self.episode_length is not None:
            max_start = max_start - self.episode_length
        # start must be >= window_size
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
        """Estimate execution price including spread and slippage.

        side: 'buy' or 'sell'
        price: mid/close price used as reference
        trade_value: absolute notional of the trade (positive)
        avg_volume: recent average volume (shares) to normalise impact

        Returns estimated execution price (per-share).
        """
        # half-spread depending on side
        half_spread = 0.5 * self.spread_pct * price
        if side == "buy":
            price_after_spread = price + half_spread
        else:
            price_after_spread = price - half_spread

        # slippage proportional to trade_size / available_volume
        # convert trade_value (currency) to trade_shares estimate by dividing by price
        trade_shares_est = max(0.0, trade_value / (price + 1e-12))
        max_fillable = max(1.0, avg_volume * self.volume_limit)
        volume_ratio = min(1.0, trade_shares_est / max_fillable)
        # slippage grows nonlinearly with volume_ratio
        slippage = self.slippage_coeff * (volume_ratio ** 1.2) * price

        if side == "buy":
            exec_price = price_after_spread * (1.0 + slippage / (price + 1e-12))
        else:
            exec_price = price_after_spread * (1.0 - slippage / (price + 1e-12))

        return float(exec_price)

    # ----------------------- core trade execution -----------------------
    def _execute_target_exposure(self, target_frac: float, price: float, avg_volume: float):
        """Adjust current position towards `target_frac` (fraction of net_worth).

        Implementation details:
        - Compute desired position value = target_frac * net_worth
        - Compute delta_value = desired - current position value
        - Respect max leverage: cap desired exposure in absolute terms
        - Convert to intended trade_shares and limit fills by volume_limit
        - Estimate execution price (spread + slippage)
        - Update cash, position_shares, cum_fees, entry_value appropriately
        - Return executed_trade_notional (signed), fees, financing (0 here)
        """
        # Cap target_frac by max_leverage
        max_allowed_frac = min(self.max_leverage, max(1.0, abs(target_frac))) * np.sign(target_frac)
        # But better: limit exposure magnitude to max_leverage
        target_frac = float(np.clip(target_frac, -self.max_leverage, self.max_leverage))

        desired_position_value = target_frac * self.net_worth
        current_position_value = self.position_shares * price
        delta_value = desired_position_value - current_position_value

        if abs(delta_value) < 1e-8:
            return 0.0, 0.0  # nothing to do

        side = "buy" if delta_value > 0 else "sell"
        intended_trade_value = abs(delta_value)

        # convert intended value to shares and cap by volume limit
        intended_shares = intended_trade_value / (price + 1e-12)
        max_fillable_shares = max(1.0, avg_volume * self.volume_limit)
        executed_shares = float(min(intended_shares, max_fillable_shares))
        executed_value = executed_shares * price

        if executed_shares <= 0:
            return 0.0, 0.0

        exec_price = self._estimate_exec_price(side=side, price=price, trade_value=executed_value, avg_volume=avg_volume)

        # fees and commissions computed on executed_value (using exec_price)
        trade_notional = executed_shares * exec_price
        commission = abs(trade_notional) * self.commission_pct + self.commission_fixed
        self.cum_fees += commission

        # update position_shares and cash
        signed_shares = executed_shares if delta_value > 0 else -executed_shares
        self.position_shares += signed_shares
        # cash effect: buys reduce cash, sells increase cash
        cash_flow = -signed_shares * exec_price
        # apply commission
        cash_flow -= commission * np.sign(cash_flow)
        self.balance += cash_flow

        # update entry_value rough accounting: compute new average entry valuation
        # (we don't track per-lot entries here; entry_value is signed position value)
        self.entry_value = self.position_shares * exec_price

        return float(np.sign(delta_value) * executed_value), float(commission)

    # ----------------------- observation & reward -----------------------
    def _get_obs(self):
        # historical window of features (exclude current_index)
        start = int(self.current_index - self.window_size)
        end = int(self.current_index - 1)
        window = self.raw_df.loc[start : end, self.feature_cols].values.copy()

        if self.normalize_observations:
            # z-score normalization
            window = (window - self._feature_means.values) / (self._feature_stds.values + 1e-12)

        # flatten
        flat_window = window.flatten()

        # portfolio vector: [cash/net_worth, position_value/net_worth, position_shares_adj]
        pv = self.position_shares * self._get_price_at(self.current_index)
        cash_frac = 0.0 if self.net_worth == 0 else self.balance / (self.net_worth + 1e-12)
        pos_frac = 0.0 if self.net_worth == 0 else pv / (self.net_worth + 1e-12)
        pos_shares_normed = self.position_shares / (self._get_avg_volume_at(self.current_index) + 1e-12)
        portfolio_vec = np.array([cash_frac, pos_frac, pos_shares_normed], dtype=np.float32)

        obs = np.concatenate([flat_window.astype(np.float32), portfolio_vec.astype(np.float32)])
        # clip to observation bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _compute_step_reward(self):
        # use log return of net worth (stable) scaled by reward_scaling
        if self.prev_net_worth <= 0:
            raw = 0.0
        else:
            raw = np.log((self.net_worth + 1e-12) / (self.prev_net_worth + 1e-12))
        # drawdown penalty if requested
        dd = 0.0
        if self.net_worth < self.max_net_worth:
            dd = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-12)
        dd_pen = self.dd_penalty_coeff * dd
        # turnover penalty: measured by absolute change in position value (approximated by recent trade)
        turnover = 0.0
        # a simple approximation: if last trade occurred, use its commission as proxy
        if len(self.trades) > 0:
            turnover = np.abs(self.trades[-1].get("notional", 0.0))
        turnover_pen = self.turnover_penalty_coeff * turnover / (self.initial_balance + 1e-12)

        reward = self.reward_scaling * raw - dd_pen - turnover_pen
        return float(reward)

    # ----------------------- public gym methods -----------------------
    def reset(self, seed=None, options=None):
        self.seed(seed)
        self._reset_internal_state()
        self._set_start_index()
        # initialize net_worth from initial balance
        self.net_worth = float(self.initial_balance)
        self.prev_net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        # set current_index at start_index
        self.current_index = int(self.start_index)
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        assert self.action_space.contains(action), f"Action {action} invalid"
        target_frac = float(action[0])

        price = self._get_price_at(self.current_index)
        avg_volume = self._get_avg_volume_at(self.current_index)

        # execute trade to move towards target exposure
        executed_notional, commission = self._execute_target_exposure(target_frac, price, avg_volume)
        # record trade
        if abs(executed_notional) > 0:
            self.trades.append({
                "index": int(self.current_index),
                "notional": float(executed_notional),
                "commission": float(commission),
                "position_shares": float(self.position_shares),
            })

        # financing (carry) for leveraged portions: simple proportional interest per step
        # if position value and net_worth indicate borrowed amount, charge financing
        position_value = self.position_shares * price
        borrowed = max(0.0, abs(position_value) - self.net_worth)
        daily_rate = self.financing_rate_annual / 252.0
        financing_cost = borrowed * daily_rate
        self.balance -= financing_cost
        self.cum_financing += financing_cost

        # recompute net worth
        self.net_worth = self.balance + position_value
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # margin check: if net_worth falls below maintenance margin -> force liquidation (terminate)
        margin_requirement = self.maintenance_margin * abs(position_value)
        if self.net_worth <= 0 or (abs(position_value) > 0 and self.net_worth < margin_requirement):
            # forced liquidation: sell everything at current price (approx execution)
            if self.position_shares != 0:
                # execute opposite trade to close
                close_value = abs(self.position_shares) * price
                close_side = "sell" if self.position_shares > 0 else "buy"
                # approximate commission
                commission_close = close_value * self.commission_pct
                self.balance += -np.sign(self.position_shares) * self.position_shares * price - commission_close
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
            # recompute net_worth
            self.net_worth = self.balance + position_value
            info = {"reason": "margin_call_or_bankrupt"}
            reward = self._compute_step_reward()
            self.prev_net_worth = self.net_worth
            self.current_index += 1
            self.terminated = True
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, float(reward), True, False, info

        # compute reward
        reward = self._compute_step_reward()

        # increment pointers
        self.prev_net_worth = self.net_worth
        self.current_index += 1
        self.step_count += 1

        # termination conditions
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


# Quick usage hint (not executed here):
# env = RealisticTradingEnv(df, feature_cols=['Close','ret','sma_10'], window_size=50)
# obs, info = env.reset()
# action = np.array([0.2], dtype=np.float32)  # target 20% long exposure
# obs, reward, done, truncated, info = env.step(action)
