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
"""  # Module docstring describing the file and its design goals.

import gymnasium as gym  # Import Gymnasium core package, aliased to `gym`.
from gymnasium import spaces  # Import spaces submodule for action/observation spaces.
import numpy as np  # Numerical computing utilities, arrays, math.
import pandas as pd  # DataFrame support for market data and features.


class TradingEnv(gym.Env):  # Define a Gym-compatible environment by subclassing gym.Env.
    """A Gymnasium environment that models many practical aspects of trading.

    Observation: a 1D numpy array combining a flattened window of features and a small
    portfolio vector. All inputs are scaled to reasonable ranges (approx -10..10)
    for numerical stability.

    Action: continuous Box(shape=(1,), low=-1.0, high=1.0) representing target exposure
    as fraction of net worth. Example: action=0.25 => target long exposure 25% of net worth.
    Negative actions indicate short exposure. Leverage is limited by max_leverage.

    Reward: log return of net_worth between steps, optionally penalized for drawdown and turnover.
    """  # Class docstring summarizing interface and semantics.

    metadata = {"render_modes": ["human"]}  # Gym metadata specifying supported render modes.

    def __init__(
        self,
        df: pd.DataFrame,  # Input price/feature DataFrame.
        feature_cols=None,  # Which columns to include as features in observations.
        window_size: int = 50,  # Length of the historical window for observations.
        initial_balance: float = 100_000.0,  # Starting cash/net worth.
        commission_pct: float = 0.0005,  # 5 bps  # Percentage commission per trade notional.
        commission_fixed: float = 0.0,  # Fixed per-trade commission.
        spread_pct: float = 0.0002,  # 2 bps round-trip (half-spread = 1 bp)  # Bid-ask spread proxy.
        slippage_coeff: float = 0.1,  # coef controlling slippage vs trade/volume  # Impact strength.
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
        super().__init__()  # Initialize base gym.Env internals.

        # Basic data checks
        required_cols = ["Open", "High", "Low", "Close", "Volume"]  # Minimal OHLCV columns.
        if not all(c in df.columns for c in required_cols):  # Ensure df has required columns.
            raise ValueError(f"DataFrame must contain columns: {required_cols}")  # Fail early if not.

        self.raw_df = df.reset_index(drop=True).copy()  # Store a clean copy (0..N-1 index).
        self.feature_cols = feature_cols if feature_cols is not None else ["Close"]  # Default features.
        self.window_size = int(window_size)  # Ensure int and store history length.

        # Economic params
        self.initial_balance = float(initial_balance)  # Starting cash and net worth.
        self.commission_pct = float(commission_pct)  # Proportional commission.
        self.commission_fixed = float(commission_fixed)  # Fixed commission.
        self.spread_pct = float(spread_pct)  # Bid-ask spread fraction of price.
        self.slippage_coeff = float(slippage_coeff)  # Slippage/impact coefficient.
        self.volume_limit = float(volume_limit)  # Fraction of average volume we can trade.
        self.max_leverage = float(max_leverage)  # Max |position_value| / net_worth.
        self.maintenance_margin = float(maintenance_margin)  # Maintenance margin fraction.
        self.financing_rate_annual = float(financing_rate_annual)  # Annual borrow rate.
        self.reward_scaling = float(reward_scaling)  # Reward scale.
        self.dd_penalty_coeff = float(dd_penalty_coeff)  # Drawdown penalty weight.
        self.turnover_penalty_coeff = float(turnover_penalty_coeff)  # Turnover penalty weight.

        # Episode mechanics
        self.random_start = bool(random_start)  # Randomize start if True.
        self.episode_length = episode_length  # None => until end of df; else fixed horizon.

        # Precompute a few helpful series
        # We will use a short moving average of volume as a proxy for available liquidity
        self.raw_df["avg_volume"] = self.raw_df["Volume"].rolling(window=20, min_periods=1).mean()  # 20-step mean volume.

        # Normalization helpers: compute mean/std per feature over full dataset
        self.normalize_observations = bool(normalize_observations)  # Flag for z-scoring features.
        if self.normalize_observations:  # Compute per-feature stats for normalization.
            self._feature_means = self.raw_df[self.feature_cols].mean()  # Column means.
            self._feature_stds = self.raw_df[self.feature_cols].std().replace(0, 1.0)  # Std, avoid zeros by replacing with 1.
        else:  # If not normalizing, set means=0, stds=1 so features pass through.
            self._feature_means = pd.Series(0.0, index=self.feature_cols)
            self._feature_stds = pd.Series(1.0, index=self.feature_cols)

        # Observation space: flattened window + portfolio vector
        self.n_features = len(self.feature_cols)  # Number of feature columns.
        self.obs_window_shape = (self.window_size, self.n_features)  # 2D window shape (time, features).
        # we'll flatten the window; then append [cash/net_worth, position_value/net_worth, position_fraction]
        flat_len = self.window_size * self.n_features + 3  # Flattened window length plus 3 portfolio stats.
        # Bound observations to a conservative range for numerical stability
        obs_low = -10.0 * np.ones(flat_len, dtype=np.float32)  # Lower bound vector for observation Box.
        obs_high = 10.0 * np.ones(flat_len, dtype=np.float32)  # Upper bound vector for observation Box.
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)  # Define observation space.

        # Action space: continuous target exposure fraction in [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)  # Single-float action.

        # Internal state
        self._reset_internal_state()  # Initialize per-episode state variables.

    # ----------------------- internal state & helpers -----------------------
    def _reset_internal_state(self):  # Reset all per-episode variables. Called by reset().
        """Reset all per-episode variables. Called by reset()."""
        self.step_count = 0  # Number of steps taken so far in this episode.
        # position expressed as signed number of shares (positive long, negative short)
        self.position_shares = 0.0  # Current position in shares, sign indicates direction.
        self.entry_value = 0.0  # signed value of position at entry (approximate avg entry * shares).
        self.balance = float(self.initial_balance)  # cash position / free cash.
        self.net_worth = float(self.initial_balance)  # balance + mark-to-market position value.
        self.gross_exposure = 0.0  # |position_value| / net_worth (not continuously maintained elsewhere).
        self.trades = []  # List of trade records for this episode.
        self.max_net_worth = self.initial_balance  # Running peak NW for drawdown calc.
        self.prev_net_worth = self.initial_balance  # Last step's NW for log-return reward.
        self.start_index = self.window_size  # Will be overwritten in _set_start_index().
        self.current_index = self.start_index  # Pointer to current time index in df.
        self.terminated = False  # Whether we ended due to margin/bankruptcy.
        self.cum_fees = 0.0  # Accumulate commissions/fees.
        self.cum_financing = 0.0  # Accumulate financing (carry) costs.

    def seed(self, seed=None):  # Optional seeding for reproducibility.
        np.random.seed(seed)  # Set NumPy RNG seed.

    # ----------------------- data helpers -----------------------
    def _set_start_index(self):  # Choose the episode start index.
        """Randomize or set the start index for the episode.

        Ensures there is enough room for `window_size` history and `episode_length` future.
        """
        max_start = len(self.raw_df) - 1  # Last valid index (exclusive of final step logic).
        if self.episode_length is not None:  # If fixed episode length, constrain start to leave room.
            max_start = max_start - self.episode_length
        # start must be >= window_size
        min_start = self.window_size  # Must have at least `window_size` points of history.
        if min_start >= len(self.raw_df):  # Guard against too-short datasets.
            raise ValueError("Data too short for chosen window_size")
        if self.random_start:  # Randomize within [min_start, max_start].
            self.start_index = np.random.randint(min_start, max_start + 1)
        else:  # Deterministic start right after the initial window.
            self.start_index = min_start
        self.current_index = int(self.start_index)  # Sync current index to chosen start.

    def _get_price_at(self, idx: int) -> float:  # Helper: fetch Close price at index.
        return float(self.raw_df.loc[idx, "Close"])  # Cast to float for numerical ops.

    def _get_avg_volume_at(self, idx: int) -> float:  # Helper: fetch rolling avg volume.
        return float(self.raw_df.loc[idx, "avg_volume"])  # Precomputed 20-step average.

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
        half_spread = 0.5 * self.spread_pct * price  # Half of bid-ask spread in price units.
        if side == "buy":  # Buyers pay ask => add half-spread.
            price_after_spread = price + half_spread
        else:  # Sellers hit bid => subtract half-spread.
            price_after_spread = price - half_spread

        # slippage proportional to trade_size / available_volume
        # convert trade_value (currency) to trade_shares estimate by dividing by price
        trade_shares_est = max(0.0, trade_value / (price + 1e-12))  # Notional -> shares; avoid divide-by-zero.
        max_fillable = max(1.0, avg_volume * self.volume_limit)  # Cap we can consume this step.
        volume_ratio = min(1.0, trade_shares_est / max_fillable)  # Fraction of cap we try to trade.
        # slippage grows nonlinearly with volume_ratio
        slippage = self.slippage_coeff * (volume_ratio ** 1.2) * price  # Nonlinear impact scaled by price.

        if side == "buy":  # Impact worsens price for buy.
            exec_price = price_after_spread * (1.0 + slippage / (price + 1e-12))
        else:  # Impact worsens price for sell (lower).
            exec_price = price_after_spread * (1.0 - slippage / (price + 1e-12))

        return float(exec_price)  # Return the estimated per-share execution price.

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
        max_allowed_frac = min(self.max_leverage, max(1.0, abs(target_frac))) * np.sign(target_frac)  # NOTE: computed but UNUSED.
        # But better: limit exposure magnitude to max_leverage
        target_frac = float(np.clip(target_frac, -self.max_leverage, self.max_leverage))  # Enforce leverage bound.

        desired_position_value = target_frac * self.net_worth  # Desired $ exposure.
        current_position_value = self.position_shares * price  # Current $ exposure.
        delta_value = desired_position_value - current_position_value  # How much notional to adjust.

        if abs(delta_value) < 1e-8:  # If effectively at target, do nothing.
            return 0.0, 0.0  # nothing to do

        side = "buy" if delta_value > 0 else "sell"  # Direction needed to reach target.
        intended_trade_value = abs(delta_value)  # Absolute notional to trade.

        # convert intended value to shares and cap by volume limit
        intended_shares = intended_trade_value / (price + 1e-12)  # Notional -> shares.
        max_fillable_shares = max(1.0, avg_volume * self.volume_limit)  # Liquidity cap in shares.
        executed_shares = float(min(intended_shares, max_fillable_shares))  # Apply partial fill if needed.
        executed_value = executed_shares * price  # Executed notional at reference price.

        if executed_shares <= 0:  # Nothing filled -> no state change.
            return 0.0, 0.0

        exec_price = self._estimate_exec_price(side=side, price=price, trade_value=executed_value, avg_volume=avg_volume)  # Price with spread+impact.

        # fees and commissions computed on executed_value (using exec_price)
        trade_notional = executed_shares * exec_price  # Actual notional at exec price.
        commission = abs(trade_notional) * self.commission_pct + self.commission_fixed  # Commission calc.
        self.cum_fees += commission  # Accumulate fees.

        # update position_shares and cash
        signed_shares = executed_shares if delta_value > 0 else -executed_shares  # Positive for buy, negative for sell.
        self.position_shares += signed_shares  # Update position.
        # cash effect: buys reduce cash, sells increase cash
        cash_flow = -signed_shares * exec_price  # Sell => +cash, Buy => -cash.
        # apply commission
        cash_flow -= commission * np.sign(cash_flow)  # WARNING: BUG for buys — commission is ADDED instead of subtracted.
        # Correct would be: cash_flow -= commission  # always reduce cash by commission regardless of side.
        self.balance += cash_flow  # Update cash balance.

        # update entry_value rough accounting: compute new average entry valuation
        # (we don't track per-lot entries here; entry_value is signed position value)
        self.entry_value = self.position_shares * exec_price  # Approximate average entry valuation.

        return float(np.sign(delta_value) * executed_value), float(commission)  # Return signed notional (ref price) and fees.

    # ----------------------- observation & reward -----------------------
    def _get_obs(self):  # Build the observation vector for the agent.
        # historical window of features (exclude current_index)
        start = int(self.current_index - self.window_size)  # Start of window (inclusive).
        end = int(self.current_index - 1)  # End of window (inclusive), excludes current tick.
        window = self.raw_df.loc[start : end, self.feature_cols].values.copy()  # Slice features window.

        if self.normalize_observations:  # Apply z-score normalization if enabled.
            # z-score normalization
            window = (window - self._feature_means.values) / (self._feature_stds.values + 1e-12)  # Broadcasted z-score.

        # flatten
        flat_window = window.flatten()  # Convert (W, F) to (W*F,).

        # portfolio vector: [cash/net_worth, position_value/net_worth, position_shares_adj]
        pv = self.position_shares * self._get_price_at(self.current_index)  # Current position value in $.
        cash_frac = 0.0 if self.net_worth == 0 else self.balance / (self.net_worth + 1e-12)  # Cash as fraction of NW.
        pos_frac = 0.0 if self.net_worth == 0 else pv / (self.net_worth + 1e-12)  # Position value as fraction of NW.
        pos_shares_normed = self.position_shares / (self._get_avg_volume_at(self.current_index) + 1e-12)  # Position scaled by liquidity.
        portfolio_vec = np.array([cash_frac, pos_frac, pos_shares_normed], dtype=np.float32)  # 3-element portfolio state.

        obs = np.concatenate([flat_window.astype(np.float32), portfolio_vec.astype(np.float32)])  # Final observation.
        # clip to observation bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)  # Enforce Box bounds [-10, 10].
        return obs  # Return observation vector.

    def _compute_step_reward(self):  # Compute reward for the last transition.
        # use log return of net worth (stable) scaled by reward_scaling
        if self.prev_net_worth <= 0:  # Avoid log issues or undefined prior NW.
            raw = 0.0
        else:
            raw = np.log((self.net_worth + 1e-12) / (self.prev_net_worth + 1e-12))  # Log NW growth since previous step.
        # drawdown penalty if requested
        dd = 0.0
        if self.net_worth < self.max_net_worth:  # If under water vs peak, compute drawdown fraction.
            dd = (self.max_net_worth - self.net_worth) / (self.max_net_worth + 1e-12)
        dd_pen = self.dd_penalty_coeff * dd  # Scale drawdown penalty.
        # turnover penalty: measured by absolute change in position value (approximated by recent trade)
        turnover = 0.0
        # a simple approximation: if last trade occurred, use its commission as proxy
        if len(self.trades) > 0:  # If any trade happened this step, approximate turnover by notional traded.
            turnover = np.abs(self.trades[-1].get("notional", 0.0))
        turnover_pen = self.turnover_penalty_coeff * turnover / (self.initial_balance + 1e-12)  # Normalize by initial NW.

        reward = self.reward_scaling * raw - dd_pen - turnover_pen  # Combine components.
        return float(reward)  # Return scalar reward.

    # ----------------------- public gym methods -----------------------
    def reset(self, seed=None, options=None):  # Standard Gym reset.
        self.seed(seed)  # Seed RNG if provided.
        self._reset_internal_state()  # Zero/initialize per-episode state.
        self._set_start_index()  # Choose start index (random/deterministic).
        # initialize net_worth from initial balance
        self.net_worth = float(self.initial_balance)  # Reset NW to starting cash.
        self.prev_net_worth = float(self.initial_balance)  # Reset previous NW for reward calc.
        self.max_net_worth = float(self.initial_balance)  # Reset peak NW.
        # set current_index at start_index
        self.current_index = int(self.start_index)  # Ensure current index equals start.
        obs = self._get_obs()  # Build initial observation.
        return obs, {}  # Gymnasium API: return (obs, info) on reset.

    def step(self, action):  # Standard Gym step.
        assert self.action_space.contains(action), f"Action {action} invalid"  # Validate action shape/range.
        target_frac = float(action[0])  # Extract scalar target exposure fraction.

        price = self._get_price_at(self.current_index)  # Reference price for this step.
        avg_volume = self._get_avg_volume_at(self.current_index)  # Liquidity proxy for this step.

        # execute trade to move towards target exposure
        executed_notional, commission = self._execute_target_exposure(target_frac, price, avg_volume)  # Try to reach target.
        # record trade
        if abs(executed_notional) > 0:  # Log a trade if anything executed.
            self.trades.append({
                "index": int(self.current_index),  # Time index traded.
                "notional": float(executed_notional),  # Signed notional at reference price.
                "commission": float(commission),  # Fees paid for the trade.
                "position_shares": float(self.position_shares),  # Position after trade.
            })

        # financing (carry) for leveraged portions: simple proportional interest per step
        # if position value and net_worth indicate borrowed amount, charge financing
        position_value = self.position_shares * price  # Mark-to-market position value.
        borrowed = max(0.0, abs(position_value) - self.net_worth)  # Amount of exposure beyond equity.
        daily_rate = self.financing_rate_annual / 252.0  # Approx daily interest rate.
        financing_cost = borrowed * daily_rate  # Cost for carrying leverage over one step (assumed daily).
        self.balance -= financing_cost  # Deduct financing from cash.
        self.cum_financing += financing_cost  # Track cumulative financing costs.

        # recompute net worth
        self.net_worth = self.balance + position_value  # Update NW = cash + position value.
        self.max_net_worth = max(self.max_net_worth, self.net_worth)  # Update peak NW if new high.

        # margin check: if net_worth falls below maintenance margin -> force liquidation (terminate)
        margin_requirement = self.maintenance_margin * abs(position_value)  # Required equity vs exposure.
        if self.net_worth <= 0 or (abs(position_value) > 0 and self.net_worth < margin_requirement):  # Trigger liquidation.
            # forced liquidation: sell everything at current price (approx execution)
            if self.position_shares != 0:  # Only if there is an open position.
                # execute opposite trade to close
                close_value = abs(self.position_shares) * price  # Notional to close at ref price.
                close_side = "sell" if self.position_shares > 0 else "buy"  # (Unused) side of closing trade.
                # approximate commission
                commission_close = close_value * self.commission_pct  # Use proportional fee (ignores fixed).
                # Apply cash effect of closing position at reference price
                self.balance += -np.sign(self.position_shares) * self.position_shares * price - commission_close  # BUG: sign error causes wrong cash update.
                # Correct would be: cash_flow = -(-self.position_shares) * price; self.balance += cash_flow - commission_close
                self.cum_fees += commission_close  # Accumulate close-out commission.
                self.trades.append({
                    "index": int(self.current_index),  # Time of liquidation.
                    "notional": float(-np.sign(self.position_shares) * close_value),  # Signed notional closed.
                    "commission": float(commission_close),  # Fees on liquidation.
                    "position_shares": 0.0,  # Position is now flat.
                    "liquidation": True,  # Flag forced liquidation.
                })
                self.position_shares = 0.0  # Reset position.
                position_value = 0.0  # No exposure remains.
            # recompute net_worth
            self.net_worth = self.balance + position_value  # NW after forced close.
            info = {"reason": "margin_call_or_bankrupt"}  # Info dict indicating cause of termination.
            reward = self._compute_step_reward()  # Compute reward for this final transition.
            self.prev_net_worth = self.net_worth  # Update prev NW.
            self.current_index += 1  # Advance time index once more.
            self.terminated = True  # Mark terminated state.
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)  # Return zero observation after termination.
            return obs, float(reward), True, False, info  # Gym step return: (obs, reward, terminated, truncated, info)

        # compute reward
        reward = self._compute_step_reward()  # Normal step reward.

        # increment pointers
        self.prev_net_worth = self.net_worth  # Save NW for next-step reward calc.
        self.current_index += 1  # Move to next time index.
        self.step_count += 1  # Increment step counter.

        # termination conditions
        done = False  # Default: episode continues.
        if self.episode_length is not None and (self.current_index - self.start_index) >= self.episode_length:  # Length cap.
            done = True
        if self.current_index >= len(self.raw_df) - 1:  # If we reached (or passed) end of data, end episode.
            done = True

        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)  # Next observation or zeros.

        info = {
            "net_worth": float(self.net_worth),  # Current NW.
            "cum_fees": float(self.cum_fees),  # Total commissions so far.
            "cum_financing": float(self.cum_financing),  # Total financing costs so far.
            "position_shares": float(self.position_shares),  # Current position size.
            "start_index": int(self.start_index),  # Episode start index (for analysis).
        }

        return obs, float(reward), bool(done), False, info  # Standard Gymnasium 0.27 step signature.

    def render(self, mode="human"):  # Simple text render.
        price = self._get_price_at(self.current_index)  # Get current price.
        print(
            f"Idx:{self.current_index} Price:{price:.4f} NW:{self.net_worth:.2f} Bal:{self.balance:.2f} PosShares:{self.position_shares:.4f}"
        )  # Print a compact snapshot of state.


# Quick usage hint (not executed here):  # Example of how to instantiate and step the env.
# env = RealisticTradingEnv(df, feature_cols=['Close','ret','sma_10'], window_size=50)
# obs, info = env.reset()
# action = np.array([0.2], dtype=np.float32)  # target 20% long exposure
# obs, reward, done, truncated, info = env.step(action)


# -------------------------
# Key issues & suggested fixes (inline notes)
# -------------------------
# 1) _execute_target_exposure: the variable `max_allowed_frac` is computed but never used. You can remove it, or
#    incorporate it into the target exposure logic. Current clipping with np.clip is sufficient.
#
# 2) Commission cashflow sign bug on buys:
#      cash_flow -= commission * np.sign(cash_flow)
#    For buys (cash_flow < 0), this ADDS commission (makes outflow smaller). Should always reduce cash:
#      cash_flow -= commission
#
# 3) Liquidation cash update sign bug:
#      self.balance += -np.sign(self.position_shares) * self.position_shares * price - commission_close
#    For a long position, this subtracts proceeds instead of adding them; for a short it also has the wrong sign.
#    Correct approach mirrors the generic trade cashflow logic. Example fix:
#      close_shares = -self.position_shares
#      cash_flow = -close_shares * price  # sell long => +, buy to cover short => -
#      self.balance += cash_flow - commission_close
#      self.position_shares = 0.0
#
# 4) Turnover penalty uses last trade *notional* as a proxy. This is simple but may be noisy with partial fills.
#    Consider using absolute change in position value or cumulative traded notional over the step.
#
# 5) Financing assumes daily steps (252 trading days). If your data is not daily, adjust the divisor accordingly
#    (e.g., intraday minutes -> 252*X per year).
#
# 6) Observation clipping to [-10, 10] can saturate signals if features are poorly scaled. Ensure feature engineering
#    and z-scoring produce values in reasonable ranges.
#
# 7) `gross_exposure` is initialized but never updated elsewhere. Either maintain it each step or remove it.
#
# 8) In liquidation, `close_side` is computed but unused; either use it to estimate execution price with spread/impact
#    for more realism, or remove it.
