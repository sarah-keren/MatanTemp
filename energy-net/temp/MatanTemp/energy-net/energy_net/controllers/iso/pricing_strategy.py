"""
Pricing Strategy Module for ISO Controllers

This module implements the Strategy pattern for different pricing policies used by the
Independent System Operator (ISO). It defines a common interface for all pricing strategies
and provides concrete implementations for:

1. Online Pricing: Real-time price setting at each time step
2. Quadratic Pricing: Polynomial-based pricing with coefficients set at the beginning
3. Constant Pricing: Fixed pricing throughout the episode

Each strategy handles:
- Action space definition based on its policy
- Processing agent actions into actual prices and dispatch
- Validation of actions within price boundaries
- Day-ahead vs. real-time action processing

This design allows for easy extension to new pricing policies by implementing
additional strategy classes.
"""

from typing import Dict, Any, Union, Tuple, List, Optional
import numpy as np
import logging
from abc import ABC, abstractmethod
from energy_net.market.pricing.pricing_policy import PricingPolicy
from energy_net.defs import Bounds
from energy_net.market.iso.quadratic_pricing_iso import QuadraticPricingISO, SublinearPricingISO
from gymnasium import spaces

class PricingStrategy(ABC):
    """
    Base strategy interface for pricing policies.
    
    This abstract class defines the interface for all pricing strategies.
    Each concrete strategy handles a specific pricing policy (Quadratic, Online, Constant).
    
    The Strategy pattern allows the ISO controller to use different pricing mechanisms
    without changing its core logic, by delegating pricing decisions to the appropriate
    strategy object.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the base pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            logger: Logger instance for logging
        """
        self.min_price = min_price
        self.max_price = max_price
        self.max_steps_per_episode = max_steps_per_episode
        self.logger = logger
    
    @abstractmethod
    def create_action_space(self, use_dispatch_action: bool = False) -> Bounds:
        """
        Create the appropriate action space for this pricing strategy.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A gymnasium Space object representing the action space
        """
        pass
    
    @abstractmethod
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the pricing strategy.
        
        Args:
            action: The action taken by the agent
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value (predicted_demand if not controlled by action)
            - first_action_taken: Updated first_action_taken flag
        """
        pass


class QuadraticPricingStrategy(PricingStrategy):
    """
    Strategy for the Quadratic pricing policy.
    
    This strategy uses polynomial coefficients to determine prices. The agent sets
    coefficients for quadratic functions at the beginning of an episode (day-ahead),
    and these coefficients are then used to calculate prices throughout the day
    based on demand.
    
    Pricing Formula:
        price = a * demand² + b * demand + c
    
    The agent sets the coefficients [a, b, c] for both buy and sell prices,
    resulting in 6 total coefficients.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the quadratic pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the quadratic pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        policy_config = config.get('quadratic', {})
        dispatch_config = policy_config.get('dispatch', {})
        poly_config = policy_config.get('polynomial', {})
        
        self.dispatch_min = dispatch_config.get('min', 0.0)
        self.dispatch_max = dispatch_config.get('max', 500.0)
        self.low_poly = poly_config.get('min', -100.0)
        self.high_poly = poly_config.get('max', 100.0)
        
        # Initialize price coefficients and dispatch profile
        self.buy_coef = np.zeros(3, dtype=np.float32)   # [b0, b1, b2]
        self.sell_coef = np.zeros(3, dtype=np.float32)  # [s0, s1, s2]
        
        # Initialize ISO pricing objects
        self.buy_iso = None
        self.sell_iso = None
    
    def create_action_space(self, use_dispatch_action: bool = False) -> Bounds:
        """
        Create the action space for quadratic pricing, optionally including dispatch.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A Box space with dimensions for polynomial coefficients and optionally dispatch
        """
        if use_dispatch_action:
            # Include dispatch in the action space (now just a single value)
            low_array = np.concatenate((
                np.full(6, self.low_poly, dtype=np.float32),
                np.array([self.dispatch_min], dtype=np.float32)
            ))
            high_array = np.concatenate((
                np.full(6, self.high_poly, dtype=np.float32),
                np.array([self.dispatch_max], dtype=np.float32)
            ))
        else:
            # Only include pricing coefficients
            low_array = np.full(6, self.low_poly, dtype=np.float32)
            high_array = np.full(6, self.high_poly, dtype=np.float32)
                
        return Bounds(
            low=low_array,
            high=high_array,
            dtype=np.float32
        )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the quadratic pricing strategy.
        
        In the quadratic pricing model, price coefficients are set on the first step
        (day-ahead pricing), but dispatch is now determined at each step.
        
        Action format when use_dispatch_action is False:
            [b0, b1, b2, s0, s1, s2]
            - b0, b1, b2: Buy price polynomial coefficients
            - s0, s1, s2: Sell price polynomial coefficients
            
        Action format when use_dispatch_action is True:
            [b0, b1, b2, s0, s1, s2, dispatch]
            - The dispatch value is used for the current step
            
        Args:
            action: The action taken by the agent 
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        iso_buy_price = 0.0
        iso_sell_price = 0.0
        dispatch = predicted_demand  # Default to predicted demand if dispatch not provided
        
        action = np.array(action).flatten()
        
        # Process pricing coefficients on first step (day-ahead), validating total length
        if step_count == 1 and not first_action_taken:
            # Expect 6 coefficients + 1 dispatch value if using dispatch
            expected_length = 6 + (1 if use_dispatch_action else 0)
            if len(action) < expected_length:
                if self.logger:
                    self.logger.error(
                        f"Expected at least {expected_length} values for quadratic pricing "
                        f"(6 coeffs{' + dispatch' if use_dispatch_action else ''}), got {len(action)}"
                    )
                raise ValueError(
                    f"Expected at least {expected_length} values for quadratic pricing "
                    f"(6 coeffs{' + dispatch' if use_dispatch_action else ''}), got {len(action)}"
                )
            # Extract the 6 polynomial coefficients
            self.buy_coef = action[0:3]
            self.sell_coef = action[3:6]
            
            # Initialize ISO pricing objects
            self.buy_iso = QuadraticPricingISO(
                buy_a=float(self.buy_coef[0]),
                buy_b=float(self.buy_coef[1]),
                buy_c=float(self.buy_coef[2])
            )
            # Use sub-linear pricing for feed-in tariff
            self.sell_iso = SublinearPricingISO(
                feed_lin=float(self.sell_coef[0]),
                gamma=float(self.sell_coef[1])
            )

            first_action_taken = True
            if self.logger:
                log_msg = f"Day-ahead polynomial for BUY: {self.buy_coef}, SELL: {self.sell_coef}"
                self.logger.info(log_msg)
        
        # Calculate prices using the polynomial coefficients
        buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': predicted_demand}) if self.buy_iso else lambda x: 0
        iso_buy_price = max(buy_pricing_fn(predicted_demand), 0)

        # Sub-linear feed-in tariff pricing
        sell_pricing_fn = self.sell_iso.get_pricing_function({}) if self.sell_iso else lambda x: 0
        iso_sell_price = max(sell_pricing_fn(predicted_demand), 0)
        
        # Process dispatch at each step if enabled
        if use_dispatch_action:
            # If it's the first step, we already extracted the first 6 values above
            # For the dispatch, use the 7th value
            if len(action) >= 7:
                dispatch = action[6]
                # ENSURE STRICT CLIPPING of dispatch values
                dispatch = float(np.clip(dispatch, self.dispatch_min, self.dispatch_max))
                self.logger.info(f"Clipped dispatch to {dispatch:.2f} [range: {self.dispatch_min:.1f}-{self.dispatch_max:.1f}]")
            else:
                if self.logger:
                    self.logger.warning(f"Expected at least 7 values for action with dispatch, got {len(action)}. Using predicted demand as dispatch.")
                dispatch = predicted_demand
        
        if self.logger:
            self.logger.info(
                f"Step {step_count} - ISO Prices: Sell {iso_sell_price:.2f}, Buy {iso_buy_price:.2f}, " +
                f"Dispatch: {dispatch:.2f}"
            )
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class ConstantPricingStrategy(PricingStrategy):
    """
    Strategy for the Constant pricing policy.
    
    This strategy uses constant prices for an entire episode. The agent sets 
    fixed buy and sell prices at the beginning of an episode (day-ahead),
    and these prices remain unchanged throughout the day.
    
    This is the simplest pricing strategy and serves as a baseline for
    comparison with more dynamic strategies.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the constant pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the constant pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        policy_config = config.get('constant', {})
        dispatch_config = policy_config.get('dispatch', {})
        poly_config = policy_config.get('polynomial', {})
        
        self.dispatch_min = dispatch_config.get('min', 0.0)
        self.dispatch_max = dispatch_config.get('max', 500.0)
        self.low_const = poly_config.get('min', min_price)
        self.high_const = poly_config.get('max', max_price)
        
        # Initialize constant prices
        self.const_buy = 0.0
        self.const_sell = 0.0
        
        # Initialize ISO pricing objects
        self.buy_iso = None
        self.sell_iso = None
    
    def create_action_space(self, use_dispatch_action: bool = False) -> Bounds:
        """
        Create the action space for constant pricing, optionally including dispatch.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A Box space with dimensions for constant buy/sell prices and optionally dispatch
        """
        if use_dispatch_action:
            # Only include dispatch in the action space for constant pricing
            low_array = np.array([self.dispatch_min], dtype=np.float32)
            high_array = np.array([self.dispatch_max], dtype=np.float32)
        else:
            # Only include constant prices
            low_array = np.array([self.min_price, self.min_price], dtype=np.float32)
            high_array = np.array([self.max_price, self.max_price], dtype=np.float32)
        
        return Bounds(
            low=low_array,
            high=high_array,
            dtype=np.float32
        )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the constant pricing strategy.
        
        Prices are set on the first step and remain constant, but dispatch is now
        determined at each step.
        
        Args:
            action: The action taken by the agent (constant buy/sell prices + optional dispatch)
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        iso_buy_price = 0.0
        iso_sell_price = 0.0
        dispatch = predicted_demand  # Default to predicted demand if dispatch not provided
        
        action = np.array(action).flatten()
        
        # Process constant prices on first step (day-ahead)
        if step_count == 1 and not first_action_taken:
            if not use_dispatch_action:
                # Only extract constant prices if no dispatch action in vector
                if len(action) >= 2:
                    self.const_buy = float(action[0])
                    self.const_sell = float(action[1])
                else:
                    if self.logger:
                        self.logger.error(f"Expected at least 2 price values, got {len(action)}.")
                    raise ValueError(f"Expected at least 2 price values, got {len(action)}.")
            # When use_dispatch_action=True, prices remain as defaults from config
            # Dispatch will be processed separately below
            # Initialize ISO pricing objects for constant prices
            self.buy_iso = QuadraticPricingISO(
                buy_a=0.0, buy_b=0.0, buy_c=self.const_buy
            )
            self.sell_iso = QuadraticPricingISO(
                buy_a=0.0, buy_b=0.0, buy_c=self.const_sell
            )
            first_action_taken = True
            if self.logger:
                self.logger.info(f"Day-ahead constant prices - BUY: {self.const_buy}, SELL: {self.const_sell}")
        
        # Calculate constant prices
        buy_pricing_fn = self.buy_iso.get_pricing_function({'demand': predicted_demand}) if self.buy_iso else lambda x: 0
        iso_buy_price = buy_pricing_fn(1.0)

        sell_pricing_fn = self.sell_iso.get_pricing_function({'demand': predicted_demand}) if self.sell_iso else lambda x: 0
        iso_sell_price = sell_pricing_fn(1.0)
        
        # Process dispatch at each step if enabled
        if use_dispatch_action:
            # Single-element action represents dispatch only
            if len(action) >= 1:
                dispatch = float(np.clip(action[0], self.dispatch_min, self.dispatch_max))
                self.logger.info(f"Clipped dispatch to {dispatch:.2f} [range: {self.dispatch_min:.1f}-{self.dispatch_max:.1f}]")
            else:
                if self.logger:
                    self.logger.warning(f"Expected dispatch value, got action length {len(action)}. Using predicted demand as dispatch.")
                dispatch = predicted_demand
        
        if self.logger:
            self.logger.info(
                f"Step {step_count} - ISO Prices: Sell {iso_sell_price:.2f}, Buy {iso_buy_price:.2f}, " +
                f"Dispatch: {dispatch:.2f}"
            )
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class OnlinePricingStrategy(PricingStrategy):
    """
    Strategy for the Online pricing policy.
    
    This strategy allows the agent to update prices at each time step (real-time pricing).
    It provides the most flexibility, allowing the ISO to respond immediately to changing
    grid conditions.
    
    Action format when use_dispatch_action is False:
        [buy_price, sell_price]
        
    Action format when use_dispatch_action is True:
        [buy_price, sell_price, dispatch]
    
    Each action directly sets the prices (and optionally dispatch) for the current time step.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the online pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the online pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        # Extract specific bounds from the config
        online_config = config.get('online', {})
        self.buy_price_min = online_config.get('buy_price', {}).get('min', min_price)
        self.buy_price_max = online_config.get('buy_price', {}).get('max', max_price)
        self.sell_price_min = online_config.get('sell_price', {}).get('min', min_price)
        self.sell_price_max = online_config.get('sell_price', {}).get('max', max_price)
        self.dispatch_min = online_config.get('dispatch', {}).get('min', 0.0)
        self.dispatch_max = online_config.get('dispatch', {}).get('max', 500.0)
        
        if self.logger:
            self.logger.info(
                f"Initialized OnlinePricingStrategy with bounds: "
                f"Buy Price [{self.buy_price_min}, {self.buy_price_max}], "
                f"Sell Price [{self.sell_price_min}, {self.sell_price_max}]"
            )
    
    def create_action_space(self, use_dispatch_action: bool = False) -> Bounds:
        """
        Create the action space for online pricing, optionally including dispatch.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A Box space with dimensions for buy/sell prices and optionally dispatch
        """
        if use_dispatch_action:
            # Include dispatch in the action space
            return Bounds(
                low=np.array([self.buy_price_min, self.sell_price_min, self.dispatch_min], dtype=np.float32),
                high=np.array([self.buy_price_max, self.sell_price_max, self.dispatch_max], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Only include buy/sell prices
            return Bounds(
                low=np.array([self.buy_price_min, self.sell_price_min], dtype=np.float32),
                high=np.array([self.buy_price_max, self.sell_price_max], dtype=np.float32),
                dtype=np.float32
            )
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the online pricing strategy.
        
        In the online pricing model, actions directly set the buy/sell prices
        for the current time step, allowing for real-time price adjustments.
        
        Action format when use_dispatch_action is False:
            [buy_price, sell_price]
            
        Action format when use_dispatch_action is True:
            [buy_price, sell_price, dispatch]
            
        Args:
            action: The action taken by the agent
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current time step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price
            - sell_price: Current selling price
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        if self.logger:
            self.logger.info(f"Processing ISO action: {action}")
        
        dispatch = predicted_demand  # Default to predicted demand if dispatch not provided
        
        if isinstance(action, np.ndarray):
            action = action.flatten()
        else:
            if self.logger:
                self.logger.info(f"Converting scalar action to array: {action}")
            action = np.array([action, action])
        
        if use_dispatch_action:
            # Extract prices and dispatch from action
            if len(action) >= 3:
                # Format: [buy_price, sell_price, dispatch]
                iso_buy_price = action[0]
                iso_sell_price = action[1]
                dispatch = action[2]
            else:
                if self.logger:
                    self.logger.warning(f"Expected 3 values for action with dispatch, got {len(action)}. Using only prices.")
                iso_buy_price = action[0]
                iso_sell_price = action[1] if len(action) > 1 else action[0]
        else:
            # Extract only prices
            iso_buy_price = action[0]
            iso_sell_price = action[1] if len(action) > 1 else action[0]
            
        # Ensure the prices are within bounds
        iso_buy_price = float(np.clip(iso_buy_price, self.buy_price_min, self.buy_price_max))
        iso_sell_price = float(np.clip(iso_sell_price, self.sell_price_min, self.sell_price_max))
        
        # Ensure dispatch is within bounds if provided
        if use_dispatch_action:
            dispatch = float(np.clip(dispatch, self.dispatch_min, self.dispatch_max))
        
        if self.logger:
            log_msg = (
                f"Step {step_count} - ISO Prices: "
                f"Buy {iso_buy_price:.2f} [{self.buy_price_min}-{self.buy_price_max}], "
                f"Sell {iso_sell_price:.2f} [{self.sell_price_min}-{self.sell_price_max}]"
            )
            if use_dispatch_action:
                log_msg += f", Dispatch: {dispatch:.2f}"
            self.logger.info(log_msg)
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class IntervalPricingStrategy(PricingStrategy):
    """
    Strategy for INTERVALS pricing policy: let ISO set buy/sell prices at 3 equally spaced intervals per episode.
    """
    def __init__(self, min_price: float, max_price: float, max_steps_per_episode: int, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        policy_config = config.get('intervals', {})
        n_intervals = policy_config.get('intervals', 3)
        self.steps_per_interval = max_steps_per_episode // n_intervals
        self.current_buy = min_price
        self.current_sell = max_price
        self.last_update_step = -1

    def create_action_space(self, use_dispatch_action: bool = False) -> Bounds:
        # Only buy/sell prices (no dispatch)
        low = np.array([self.min_price, self.min_price], dtype=np.float32)
        high = np.array([self.max_price, self.max_price], dtype=np.float32)
        return Bounds(low=low, high=high, dtype=np.float32)

    def process_action(
        self,
        action: Union[float, np.ndarray, int],
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        # Update prices only at the start of each interval
        if (step_count - 1) % self.steps_per_interval == 0 and step_count != self.last_update_step:
            arr = np.array(action).flatten()
            self.current_buy = float(np.clip(arr[0], self.min_price, self.max_price))
            self.current_sell = float(np.clip(arr[1] if len(arr) > 1 else arr[0], self.min_price, self.max_price))
            self.last_update_step = step_count
        dispatch = predicted_demand
        return self.current_buy, self.current_sell, dispatch, first_action_taken


class QuadraticIntervalPricingStrategy(PricingStrategy):
    """
    Strategy for QUADRATIC_INTERVALS pricing policy: agent sets polynomial coefficients (a,b,c) for buy and (feed,gamma) for sell at each interval boundary.
    """
    def __init__(self, min_price: float, max_price: float, max_steps_per_episode: int, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        # Get polynomial bounds from quadratic config
        quad_cfg = config.get('quadratic', {})
        poly_cfg = quad_cfg.get('polynomial', {})
        self.low_poly = poly_cfg.get('min', -100.0)
        self.high_poly = poly_cfg.get('max', 100.0)
        # Number of intervals
        intervals_cfg = config.get('intervals', {})
        self.n_intervals = intervals_cfg.get('intervals', 3)
        # Steps per interval (floor division)
        self.steps_per_interval = max_steps_per_episode // self.n_intervals
        # Track current interval index
        self.current_interval = -1
        # Pricing functions per interval
        self.buy_iso = None
        self.sell_iso = None

    def create_action_space(self, use_dispatch_action: bool = False) -> Bounds:
        # Dimension: 6 coefficients per interval
        dims = self.n_intervals * 6
        low = np.full(dims, self.low_poly, dtype=np.float32)
        high = np.full(dims, self.high_poly, dtype=np.float32)
        return Bounds(low=low, high=high, dtype=np.float32)

    def process_action(
        self,
        action: Union[float, np.ndarray, int],
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        arr = np.array(action).flatten()
        # Determine which interval we're in (0-based)
        idx = min((step_count - 1) // self.steps_per_interval, self.n_intervals - 1)
        # On first step of a new interval, update coefficients
        if idx != self.current_interval:
            start = idx * 6
            buy_coef = arr[start:start+3]
            sell_coef = arr[start+3:start+6]
            # Buy: quadratic iso
            self.buy_iso = QuadraticPricingISO(
                buy_a=float(buy_coef[0]), buy_b=float(buy_coef[1]), buy_c=float(buy_coef[2])
            )
            # Sell: sublinear feed-in tariff using first two coefficients
            self.sell_iso = SublinearPricingISO(
                feed_lin=float(sell_coef[0]), gamma=float(sell_coef[1])
            )
            self.current_interval = idx
        # Compute prices based on current iso objects
        buy_fn = self.buy_iso.get_pricing_function({'demand': predicted_demand}) if self.buy_iso else (lambda x: 0)
        sell_fn = self.sell_iso.get_pricing_function({}) if self.sell_iso else (lambda x: 0)
        iso_buy_price = max(buy_fn(predicted_demand), 0)
        iso_sell_price = max(sell_fn(predicted_demand), 0)
        # Dispatch default
        dispatch = predicted_demand
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class SMPPricingStrategy(PricingStrategy):
    """
    Strategy for SMP (System Marginal Price) pricing policy.
    
    This strategy allows the ISO to set base buy/sell prices at the beginning 
    of an episode (day-ahead), with two time intervals where one interval 
    gets a discount applied to the base prices.
    
    The ISO sets the base prices once at the start, and the system automatically
    applies discounts during the specified discount hours.
    """
    
    def __init__(
        self, 
        min_price: float, 
        max_price: float,
        max_steps_per_episode: int,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SMP pricing strategy.
        
        Args:
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            config: Configuration for the SMP pricing policy
            logger: Logger instance for logging
        """
        super().__init__(min_price, max_price, max_steps_per_episode, logger)
        
        # Get SMP-specific configuration
        smp_config = config.get('smp', {})
        
        # Time intervals configuration
        intervals_config = smp_config.get('intervals', {})
        self.discount_start_hour = intervals_config.get('discount_start_hour', 10)  # Default: 10 AM
        self.discount_end_hour = intervals_config.get('discount_end_hour', 16)    # Default: 4 PM
        
        # Discount configuration
        discount_config = smp_config.get('discount', {})
        self.discount_percentage = discount_config.get('percentage', 0.2)  # Default: 20% discount
        
        # Price bounds
        price_config = smp_config.get('prices', {})
        self.buy_price_min = price_config.get('buy_min', min_price)
        self.buy_price_max = price_config.get('buy_max', max_price)
        self.sell_price_min = price_config.get('sell_min', min_price)
        self.sell_price_max = price_config.get('sell_max', max_price)
        
        # Dispatch configuration
        dispatch_config = smp_config.get('dispatch', {})
        self.dispatch_min = dispatch_config.get('min', 0.0)
        self.dispatch_max = dispatch_config.get('max', 500.0)
        
        # Base prices set by the ISO at the beginning
        self.base_buy_price = 0.0
        self.base_sell_price = 0.0
        
        # Calculate steps per hour (assuming 48 steps per day = 30 min per step)
        self.steps_per_hour = max_steps_per_episode / 24.0
        
        if self.logger:
            self.logger.info(
                f"Initialized SMP Strategy - Discount hours: {self.discount_start_hour}-{self.discount_end_hour}, "
                f"Discount: {self.discount_percentage*100:.1f}%, Steps per hour: {self.steps_per_hour:.2f}"
            )
    
    def create_action_space(self, use_dispatch_action: bool = False) -> spaces.Space:
        """
        Create the action space for SMP pricing.
        
        Args:
            use_dispatch_action: Whether to include dispatch in the action space
            
        Returns:
            A Box space with dimensions for base buy/sell prices and optionally dispatch
        """
        if use_dispatch_action:
            # Include dispatch in the action space
            return spaces.Box(
                low=np.array([self.buy_price_min, self.sell_price_min, self.dispatch_min], dtype=np.float32),
                high=np.array([self.buy_price_max, self.sell_price_max, self.dispatch_max], dtype=np.float32),
                dtype=np.float32
            )
        else:
            # Only include base buy/sell prices
            return spaces.Box(
                low=np.array([self.buy_price_min, self.sell_price_min], dtype=np.float32),
                high=np.array([self.buy_price_max, self.sell_price_max], dtype=np.float32),
                dtype=np.float32
            )
    
    def _get_current_hour(self, step_count: int) -> float:
        """
        Calculate the current hour of the day based on step count.
        
        Args:
            step_count: Current step in the episode (1-based)
            
        Returns:
            Current hour of the day (0-24)
        """
        # Convert 1-based step to 0-based, then to hour
        return ((step_count - 1) / self.steps_per_hour) % 24.0
    
    def _is_discount_period(self, step_count: int) -> bool:
        """
        Check if the current step is within the discount time period.
        
        Args:
            step_count: Current step in the episode
            
        Returns:
            True if in discount period, False otherwise
        """
        current_hour = self._get_current_hour(step_count)
        
        # Handle case where discount period spans midnight
        if self.discount_start_hour <= self.discount_end_hour:
            # Normal case: e.g., 10 AM to 4 PM
            return self.discount_start_hour <= current_hour < self.discount_end_hour
        else:
            # Spans midnight: e.g., 10 PM to 6 AM
            return current_hour >= self.discount_start_hour or current_hour < self.discount_end_hour
    
    def process_action(
        self, 
        action: Union[float, np.ndarray, int], 
        step_count: int,
        first_action_taken: bool,
        predicted_demand: float = 0.0,
        use_dispatch_action: bool = False
    ) -> Tuple[float, float, float, bool]:
        """
        Process the agent's action according to the SMP pricing strategy.
        
        Base prices are set on the first step (day-ahead), then discounts are 
        applied automatically during specified time intervals.
        
        Action format when use_dispatch_action is False:
            [base_buy_price, base_sell_price]
            
        Action format when use_dispatch_action is True:
            [base_buy_price, base_sell_price, dispatch]
            
        Args:
            action: The action taken by the agent
            step_count: The current step count in the episode
            first_action_taken: Whether the first action has been taken
            predicted_demand: The predicted demand for the current step
            use_dispatch_action: Whether dispatch is included in the action
            
        Returns:
            Tuple containing:
            - buy_price: Current buying price (with discount applied if in discount period)
            - sell_price: Current selling price (with discount applied if in discount period)
            - dispatch: Current dispatch value
            - first_action_taken: Updated first_action_taken flag
        """
        dispatch = predicted_demand  # Default to predicted demand if dispatch not provided
        
        action = np.array(action).flatten()
        
        # Set base prices on first step (day-ahead)
        if step_count == 1 and not first_action_taken:
            if len(action) >= 2:
                self.base_buy_price = float(np.clip(action[0], self.buy_price_min, self.buy_price_max))
                self.base_sell_price = float(np.clip(action[1], self.sell_price_min, self.sell_price_max))
                first_action_taken = True
                
                if self.logger:
                    self.logger.info(
                        f"SMP Day-ahead prices set - Base Buy: {self.base_buy_price:.2f}, "
                        f"Base Sell: {self.base_sell_price:.2f}"
                    )
            else:
                if self.logger:
                    self.logger.error(f"Expected at least 2 values for SMP pricing, got {len(action)}")
                raise ValueError(f"Expected at least 2 values for SMP pricing, got {len(action)}")
        
        # Calculate current prices (apply discount if in discount period)
        is_discount_period = self._is_discount_period(step_count)
        
        if is_discount_period:
            # Apply discount to both buy and sell prices
            iso_buy_price = self.base_buy_price * (1 - self.discount_percentage)
            iso_sell_price = self.base_sell_price * (1 - self.discount_percentage)
            price_type = "DISCOUNT"
        else:
            # Use full base prices
            iso_buy_price = self.base_buy_price
            iso_sell_price = self.base_sell_price
            price_type = "FULL"
        
        # Process dispatch if enabled
        if use_dispatch_action and len(action) >= 3:
            dispatch = float(np.clip(action[2], self.dispatch_min, self.dispatch_max))
        
        # Ensure prices are non-negative
        iso_buy_price = max(iso_buy_price, 0.0)
        iso_sell_price = max(iso_sell_price, 0.0)
        
        if self.logger:
            current_hour = self._get_current_hour(step_count)
            self.logger.info(
                f"Step {step_count} (Hour {current_hour:.1f}) - SMP {price_type} Prices: "
                f"Buy {iso_buy_price:.2f}, Sell {iso_sell_price:.2f}, Dispatch: {dispatch:.2f}"
            )
        
        return iso_buy_price, iso_sell_price, dispatch, first_action_taken


class PricingStrategyFactory:
    """
    Factory class for creating pricing strategy instances.
    
    This factory implements the Factory pattern to create the appropriate
    pricing strategy based on the pricing policy enum value. It encapsulates
    the object creation logic and provides a clean interface for creating
    strategy objects.
    """
    
    @staticmethod
    def create_strategy(
        pricing_policy: PricingPolicy,
        min_price: float,
        max_price: float,
        max_steps_per_episode: int,
        action_spaces_config: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ) -> PricingStrategy:
        """
        Create the appropriate pricing strategy based on the pricing policy.
        
        Args:
            pricing_policy: The pricing policy enum value
            min_price: Minimum price boundary
            max_price: Maximum price boundary
            max_steps_per_episode: Maximum number of steps per episode
            action_spaces_config: Configuration for action spaces
            logger: Logger instance for logging
            
        Returns:
            An instance of the appropriate pricing strategy
            
        Raises:
            ValueError: If the pricing policy is not supported
        """
        if pricing_policy == PricingPolicy.QUADRATIC:
            return QuadraticPricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        elif pricing_policy == PricingPolicy.CONSTANT:
            return ConstantPricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        elif pricing_policy == PricingPolicy.ONLINE:
            return OnlinePricingStrategy(
                min_price, 
                max_price, 
                max_steps_per_episode, 
                action_spaces_config,
                logger
            )
        elif pricing_policy == PricingPolicy.INTERVALS:
            return IntervalPricingStrategy(min_price, max_price, max_steps_per_episode, action_spaces_config, logger)
        elif pricing_policy == PricingPolicy.QUADRATIC_INTERVALS:
            return QuadraticIntervalPricingStrategy(min_price, max_price, max_steps_per_episode, action_spaces_config, logger)
        elif pricing_policy == PricingPolicy.SMP:
            return SMPPricingStrategy(min_price, max_price, max_steps_per_episode, action_spaces_config, logger)
        else:
            if logger:
                logger.error(f"Unsupported pricing policy: {pricing_policy}")
            raise ValueError(f"Unsupported pricing policy: {pricing_policy}")