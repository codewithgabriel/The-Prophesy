import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any, List


class TradingEnv(gym.Env):
    """
    A realistic single-asset trading environment with execution frictions, margin,
    financing, and penalties.

    Observation
    -----------
    Flattened window of selected features (z-scored if enabled) + a portfolio vector:
        [cash_frac, pos_frac, leverage_used, unrealized_pnl_frac]
    where:
        cash_frac            = cash / net_worth
        pos_frac             = position_value / net_worth
        leverage_used        = gross_exposure / (net_worth + 1e-12)
        unrealized_pnl_frac  = unrealized_pnl / (net_worth + 1e-12)

    Action
    ------
    Box(shape=(1,), low=-1, high=1). Value is target exposure as a fraction of net worth
    (can be short if negative). Capped by max_leverage. The environment trades towards
    the target, subject to a volume limit.

    Reward
    ------
    log(NW_t / NW_{t-1}) - drawdown_penalty - turnover_penalty - financing_penalty,
    scaled by reward_scaling.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        window_size: int = 50,
        initial_balance: float = 100_000.0,
        commission_pct: float = 0.0005,      # 5 bps per notional
        commission_fixed: float = 0.0,       # fixed fee per trade
        spread_pct: float = 0.0002,          # round-trip spread; half-spread applied each side
        slippage_coeff: float = 0.1,         # slippage intensity vs volume utilization
        volume_limit: float = 0.1,           # max fraction of avg volume per step we can consume
        max_leverage: float = 3.0,           # cap on |gross_exposure / net_worth|
        maintenance_margin: float = 0.25,    # equity must be >= mm * gross_exposure
        financing_rate_annual: float = 0.02, # annualized rate on borrowed capital
        reward_scaling: float = 1.0,
        dd_penalty_coeff: float = 0.0,
        turnover_penalty_coeff: float = 0.0, # per-step: |trade_notional| / initial_balance
        normalize_observations: bool = True,
        random_start: bool = True,
        episode_length: Optional[int] = None,
    ):
        super().__init__()

        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        if not all(c in df.columns for c in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")

        self.raw_df = df.reset_index(drop=True).copy()
        self.feature_cols = feature_cols if feature_cols is not None else ["Close"]
        self.window_size = int(window_size)

        # Economics/config
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

        self.random_start = bool(random_start)
        self.episode_length = episode_length if episode_length is None else int(episode_length)

        # Precompute average volume (for liquidity/slippage)
        self.raw_df["avg_volume"] = self.raw_df["Volume"].rolling(window=20, min_periods=1).mean()

        # Normalization stats
        self.normalize_observations = bool(normalize_observations)
        if self.normalize_observations:
            self._feature_means = self.raw_df[self.feature_cols].mean()
            self._feature_stds = self.raw_df[self.feature_cols].std().replace(0, 1.0)
        else:
            self._feature_means = pd.Series(0.0, index=self.feature_cols)
            self._feature_stds = pd.Series(1.0, index=self.feature_cols)

        # Observation space: flattened window + 4 portfolio stats
        self.n_features = len(self.feature_cols)
        self.obs_window_shape = (self.window_size, self.n_features)
        flat_len = self.window_size * self.n_features + 4
        obs_low = -10.0 * np.ones(flat_len, dtype=np.float32)
        obs_high = 10.0 * np.ones(flat_len, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Action space: target exposure fraction in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Randomness
        self._rng = np.random.RandomState(None)

        # Internal state
        self._reset_internal_state()

    # ------------------------------- RNG/seed --------------------------------
    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)

    # ------------------------------ State Reset ------------------------------
    def _reset_internal_state(self):
        self.step_count = 0
        self.position_shares = 0.0
        self.avg_entry_price = 0.0  # average entry price for current open position
        self.balance = float(self.initial_balance)
        self.net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self.prev_net_worth = float(self.initial_balance)
        self.start_index = self.window_size
        self.current_index = self.start_index
        self.terminated = False

        # Accounting
        self.cum_fees = 0.0
        self.cum_financing = 0.0
        self.cum_turnover_notional = 0.0
        self.cum_realized_pnl = 0.0

        # Trade log
        self.trades: List[Dict[str, Any]] = []

        # For reward shaping (most recent financing cost)
        self._last_financing_cost = 0.0
        self._last_trade_notional = 0.0

    # ------------------------------- Indexing --------------------------------
    def _set_start_index(self):
        min_start = self.window_size
        if min_start >= len(self.raw_df):
            raise ValueError("Data too short for chosen window_size")

        if self.episode_length is None:
            max_start = len(self.raw_df) - 2  # leave at least one step
        else:
            max_start = len(self.raw_df) - self.episode_length - 1

        max_start = max(min_start, max_start)
        if self.random_start:
            self.start_index = int(self._rng.randint(min_start, max_start + 1))
        else:
            self.start_index = int(min_start)

        # Guard if episode_length would run off the dataset
        if self.episode_length is not None:
            if self.start_index + self.episode_length >= len(self.raw_df):
                self.start_index = max(min_start, len(self.raw_df) - self.episode_length - 1)

        self.current_index = int(self.start_index)

    # ------------------------------- Helpers ---------------------------------
    def _get_price_at(self, idx: int) -> float:
        return float(self.raw_df.loc[idx, "Close"])

    def _get_avg_volume_at(self, idx: int) -> float:
        return float(self.raw_df.loc[idx, "avg_volume"])

    # ---------------------- Microstructure / Execution -----------------------
    def _estimate_exec_price(self, side: str, price: float, trade_value: float, avg_volume: float) -> float:
        """Apply half-spread and slippage proportional to volume utilization."""
        half_spread = 0.5 * self.spread_pct * price
        price_after_spread = price + half_spread if side == "buy" else price - half_spread

        est_shares = max(0.0, trade_value / (price + 1e-12))
        max_fillable = max(1.0, avg_volume * self.volume_limit)
        volume_ratio = min(1.0, est_shares / max_fillable)
        slippage = self.slippage_coeff * (volume_ratio ** 1.2) * price

        if side == "buy":
            exec_price = price_after_spread * (1.0 + slippage / (price + 1e-12))
        else:
            exec_price = price_after_spread * (1.0 - slippage / (price + 1e-12))
        return float(exec_price)

    def _apply_trade_accounting(
        self, side: str, exec_price: float, executed_shares: float
    ) -> Tuple[float, float, float]:
        """
        Update cash, commissions, position, avg_entry_price, realized PnL.
        Returns: (signed_shares, trade_notional, realized_pnl_from_this_trade)
        """
        assert executed_shares > 0.0
        signed_shares = executed_shares if side == "buy" else -executed_shares
        trade_notional_abs = executed_shares * exec_price
        commission = trade_notional_abs * self.commission_pct + self.commission_fixed

        # Cash flow: buy -> cash out; sell -> cash in
        if side == "buy":
            self.balance -= trade_notional_abs
        else:
            self.balance += trade_notional_abs
        self.balance -= commission
        self.cum_fees += commission

        realized_pnl = 0.0
        pos0 = self.position_shares
        pos1 = pos0 + signed_shares

        # If reducing/closing existing position (sign changes or |pos| shrinks)
        if pos0 != 0 and np.sign(pos0) != np.sign(pos1):
            # Crossing through zero: close old side first
            close_shares_abs = min(abs(pos0), executed_shares)
            realized_pnl += (exec_price - self.avg_entry_price) * close_shares_abs * np.sign(pos0)
            remaining_to_open_abs = executed_shares - close_shares_abs
            if remaining_to_open_abs > 0:
                # New side opens with fresh avg entry price = exec_price
                self.avg_entry_price = exec_price
            else:
                # Fully flat after close
                self.avg_entry_price = 0.0

        elif pos0 != 0 and np.sign(pos0) == np.sign(pos1) and abs(pos1) < abs(pos0):
            # Partial reduce without crossing zero
            close_shares_abs = abs(signed_shares)
            realized_pnl += (exec_price - self.avg_entry_price) * close_shares_abs * np.sign(pos0)
            # avg_entry_price unchanged for remaining shares

        elif pos0 != 0 and np.sign(pos0) == np.sign(pos1) and abs(pos1) > abs(pos0):
            # Increasing same-side position -> weighted average entry price
            new_abs = abs(pos1)
            self.avg_entry_price = (
                (abs(pos0) * self.avg_entry_price + abs(signed_shares) * exec_price) / (new_abs + 1e-12)
            )

        elif pos0 == 0 and pos1 != 0:
            # Opening fresh position
            self.avg_entry_price = exec_price

        self.position_shares = pos1
        self.cum_realized_pnl += realized_pnl

        self._last_trade_notional = trade_notional_abs
        self.cum_turnover_notional += trade_notional_abs

        return signed_shares, trade_notional_abs, realized_pnl

    def _execute_target_exposure(self, target_frac: float, price: float, avg_volume: float) -> Tuple[float, float, float]:
        """
        Move position towards target exposure (fraction of net worth).
        Returns: (executed_signed_shares, trade_notional_abs, realized_pnl_from_trade)
        """
        target_frac = float(np.clip(target_frac, -self.max_leverage, self.max_leverage))
        desired_position_value = target_frac * self.net_worth
        current_position_value = self.position_shares * price
        delta_value = desired_position_value - current_position_value

        if abs(delta_value) < 1e-8:
            self._last_trade_notional = 0.0
            return 0.0, 0.0, 0.0

        side = "buy" if delta_value > 0 else "sell"
        intended_value = abs(delta_value)
        intended_shares = intended_value / (price + 1e-12)

        # Liquidity cap
        max_fillable_shares = max(1.0, avg_volume * self.volume_limit)
        executed_shares = float(min(intended_shares, max_fillable_shares))
        if executed_shares <= 0.0:
            self._last_trade_notional = 0.0
            return 0.0, 0.0, 0.0

        exec_price = self._estimate_exec_price(side, price, executed_shares * price, avg_volume)
        signed_shares, trade_notional_abs, realized_pnl = self._apply_trade_accounting(
            side, exec_price, executed_shares
        )

        # Log trade
        self.trades.append(
            {
                "index": int(self.current_index),
                "side": side,
                "exec_price": float(exec_price),
                "shares": float(signed_shares),
                "notional": float(trade_notional_abs),
                "commission": float(trade_notional_abs * self.commission_pct + self.commission_fixed),
                "position_shares": float(self.position_shares),
                "realized_pnl": float(realized_pnl),
            }
        )

        return signed_shares, trade_notional_abs, realized_pnl

    # ------------------------- Observation & Reward --------------------------
    def _portfolio_metrics(self, price: float) -> Dict[str, float]:
        position_value = self.position_shares * price
        gross_exposure = abs(position_value)
        equity = self.balance + position_value  # = net worth
        unrealized_pnl = (price - self.avg_entry_price) * self.position_shares if self.position_shares != 0 else 0.0
        leverage_used = gross_exposure / (equity + 1e-12)
        return {
            "position_value": position_value,
            "gross_exposure": gross_exposure,
            "equity": equity,
            "unrealized_pnl": unrealized_pnl,
            "leverage_used": leverage_used,
        }

    def _get_obs(self) -> np.ndarray:
        start = int(self.current_index - self.window_size)
        end = int(self.current_index - 1)
        window = self.raw_df.loc[start:end, self.feature_cols].values.astype(np.float32)

        if self.normalize_observations:
            window = (window - self._feature_means.values) / (self._feature_stds.values + 1e-12)

        flat_window = window.flatten()

        price = self._get_price_at(self.current_index)
        pm = self._portfolio_metrics(price)

        cash_frac = 0.0 if pm["equity"] == 0 else self.balance / (pm["equity"] + 1e-12)
        pos_frac = 0.0 if pm["equity"] == 0 else pm["position_value"] / (pm["equity"] + 1e-12)
        unreal_frac = pm["unrealized_pnl"] / (pm["equity"] + 1e-12)

        portfolio_vec = np.array(
            [cash_frac, pos_frac, pm["leverage_used"], unreal_frac], dtype=np.float32
        )

        obs = np.concatenate([flat_window, portfolio_vec])
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs

    def _compute_step_reward(self) -> float:
        # Base log-return on net worth
        if self.prev_net_worth <= 0:
            base = 0.0
        else:
            base = np.log((self.net_worth + 1e-12) / (self.prev_net_worth + 1e-12))

        # Drawdown penalty
        dd = 0.0
        if self.net_worth < self.max_net_worth:
            dd = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-12)
        dd_pen = self.dd_penalty_coeff * dd

        # Turnover penalty (based on last trade notional)
        turnover_pen = self.turnover_penalty_coeff * (self._last_trade_notional / (self.initial_balance + 1e-12))

        # Financing penalty (shape reward by recent cost)
        financing_pen = self._last_financing_cost / (self.initial_balance + 1e-12)

        reward = self.reward_scaling * base - dd_pen - turnover_pen - financing_pen
        return float(reward)

    # ------------------------------ Gym Methods ------------------------------
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        self.seed(seed)
        self._reset_internal_state()
        self._set_start_index()
        self.net_worth = float(self.initial_balance)
        self.prev_net_worth = float(self.initial_balance)
        self.max_net_worth = float(self.initial_balance)
        self.current_index = int(self.start_index)
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        # Ensure action shape/type
        if np.isscalar(action):
            action = np.array([action], dtype=np.float32)
        action = action.astype(np.float32)
        assert self.action_space.contains(action), f"Action {action} invalid"
        target_frac = float(action[0])

        price = self._get_price_at(self.current_index)
        avg_volume = self._get_avg_volume_at(self.current_index)

        # Execute towards target
        _, trade_notional_abs, realized_pnl = self._execute_target_exposure(target_frac, price, avg_volume)

        # Financing on borrowed capital (simple model)
        pm = self._portfolio_metrics(price)
        equity = pm["equity"]
        gross = pm["gross_exposure"]

        borrowed = max(0.0, gross - equity)  # capital effectively financed
        daily_rate = self.financing_rate_annual / 252.0
        financing_cost = borrowed * daily_rate
        self.balance -= financing_cost
        self.cum_financing += financing_cost
        self._last_financing_cost = financing_cost

        # Update net worth & checks
        self.net_worth = self.balance + pm["position_value"]
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Margin call / bankruptcy: equity < maintenance_margin * gross_exposure
        margin_breach = (gross > 0) and (equity < self.maintenance_margin * gross)
        bankrupt = self.net_worth <= 0

        if margin_breach or bankrupt:
            info = {"reason": "margin_call_or_bankrupt"}

            # Force-liquidate entire position at impacted execution price
            if self.position_shares != 0.0:
                side = "sell" if self.position_shares > 0 else "buy"
                close_shares_abs = abs(self.position_shares)
                # Estimate exec price with full liquidation notional
                exec_price = self._estimate_exec_price(side, price, close_shares_abs * price, avg_volume)
                # Apply accounting as a single closing trade
                _signed, close_notional, realized_pnl_close = self._apply_trade_accounting(
                    side, exec_price, close_shares_abs
                )
                # Log liquidation trade
                self.trades.append(
                    {
                        "index": int(self.current_index),
                        "side": side,
                        "exec_price": float(exec_price),
                        "shares": float(-np.sign(side == "buy") * close_shares_abs if side == "sell" else close_shares_abs),
                        "notional": float(close_notional),
                        "commission": float(close_notional * self.commission_pct + self.commission_fixed),
                        "position_shares": float(self.position_shares),
                        "realized_pnl": float(realized_pnl_close),
                        "liquidation": True,
                    }
                )
                # Flat after liquidation
                self.position_shares = 0.0
                self.avg_entry_price = 0.0

            # Recompute portfolio after liquidation
            price = self._get_price_at(self.current_index)
            pm = self._portfolio_metrics(price)
            self.net_worth = self.balance + pm["position_value"]

            reward = self._compute_step_reward()
            self.prev_net_worth = self.net_worth
            self.current_index += 1
            self.step_count += 1
            self.terminated = True

            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, float(reward), True, False, info

        # Normal step termination logic
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
            "balance": float(self.balance),
            "position_shares": float(self.position_shares),
            "avg_entry_price": float(self.avg_entry_price),
            "cum_fees": float(self.cum_fees),
            "cum_financing": float(self.cum_financing),
            "cum_turnover_notional": float(self.cum_turnover_notional),
            "cum_realized_pnl": float(self.cum_realized_pnl),
            "start_index": int(self.start_index),
        }

        return obs, float(reward), bool(done), False, info

    # -------------------------------- Render ---------------------------------
    def render(self, mode="human"):
        price = self._get_price_at(self.current_index)
        pm = self._portfolio_metrics(price)
        print(
            f"Idx:{self.current_index} "
            f"Price:{price:.4f} "
            f"NW:{self.net_worth:.2f} "
            f"Bal:{self.balance:.2f} "
            f"Pos:{self.position_shares:.4f} "
            f"Lev:{pm['leverage_used']:.2f}"
        )
