import gym
import numpy as np
from gym import spaces

class TradingEnvironment(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost_pct=0.001, 
                 borrow_cost_pct=0.0001, margin_requirement=0.5, max_loss_pct=0.5):
        super(TradingEnvironment, self).__init__()
        self.data = data
        self.initial_balance = initial_balance
        self.transaction_cost_pct = transaction_cost_pct
        self.borrow_cost_pct = borrow_cost_pct
        self.margin_requirement = margin_requirement
        self.max_loss_pct = max_loss_pct
        
        self.reset()

        # Action space: [long_action, short_action]
        self.action_space = spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)

        # Observation space: [normalized_price, balance_ratio, long_ratio, short_ratio, 
        #                     price_ma_ratio, volatility, volume_ma_ratio]
        self.observation_space = spaces.Box(low=-10, high=10, shape=(7,), dtype=np.float32)

    def reset(self):
        self.balance = self.initial_balance
        self.long_position = 0
        self.short_position = 0
        self.current_step = 0
        self.max_portfolio_value = self.initial_balance
        return self._get_observation()

    def step(self, action):
        long_action, short_action = action
        current_price = self.data[self.current_step]['close']
        
        # Calculate portfolio value before action
        portfolio_value_before = self.balance + (self.long_position - self.short_position) * current_price
        
        # Execute long trade
        long_trade_amount = long_action * self.balance
        if long_trade_amount > 0:
            shares_bought = long_trade_amount / current_price
            cost = long_trade_amount * (1 + self.transaction_cost_pct)
            if cost <= self.balance:
                self.long_position += shares_bought
                self.balance -= cost

        # Execute short trade
        short_trade_amount = short_action * self.balance
        if short_trade_amount > 0:
            if self.balance >= short_trade_amount * self.margin_requirement:
                shares_to_short = short_trade_amount / current_price
                self.short_position += shares_to_short
                proceeds = short_trade_amount * (1 - self.transaction_cost_pct)
                self.balance += proceeds

        # Apply borrowing cost for short positions
        self.balance -= self.short_position * current_price * self.borrow_cost_pct

        # Move to next step
        self.current_step += 1
        
        # Calculate new portfolio value
        new_price = self.data[self.current_step]['close'] if self.current_step < len(self.data) else self.data[-1]['close']
        new_portfolio_value = self.balance + (self.long_position - self.short_position) * new_price
        
        # Calculate reward
        reward = (new_portfolio_value / portfolio_value_before) - 1
        
        # Update max portfolio value
        self.max_portfolio_value = max(self.max_portfolio_value, new_portfolio_value)
        
        # Check for bankruptcy or max loss
        done = new_portfolio_value <= 0 or new_portfolio_value <= self.initial_balance * (1 - self.max_loss_pct)
        
        # Check if episode is done
        if not done:
            done = self.current_step >= len(self.data) - 1

        obs = self._get_observation()
        
        # Add logging
        print(f"Step: {self.current_step}, Action: {action}, Observation: {obs}, Reward: {reward}, Done: {done}")
        print(f"Balance: {self.balance}, Long: {self.long_position}, Short: {self.short_position}, Portfolio Value: {new_portfolio_value}")
        
        return obs, reward, done, {}

    def _get_observation(self):
        price_data = self.data[max(0, self.current_step-30):self.current_step+1]
        current_price = price_data[-1]['close']
        
        # Calculate indicators
        ma_short = np.mean([x['close'] for x in price_data[-5:]])
        ma_long = np.mean([x['close'] for x in price_data[-20:]])
        volatility = np.std([x['close'] for x in price_data])
        volume = price_data[-1]['volume']
        volume_ma = np.mean([x['volume'] for x in price_data[-5:]])

        portfolio_value = self.balance + (self.long_position - self.short_position) * current_price
        
        price_mean = np.mean([x['close'] for x in price_data])
        price_std = np.std([x['close'] for x in price_data])
        normalized_price = (current_price - price_mean) / (price_std + 1e-8)
        balance_ratio = self.balance / (portfolio_value + 1e-8)
        long_ratio = (self.long_position * current_price) / (portfolio_value + 1e-8)
        short_ratio = (self.short_position * current_price) / (portfolio_value + 1e-8)
        price_ma_ratio = ma_short / (ma_long + 1e-8)
        volume_ma_ratio = volume / (volume_ma + 1e-8)
        
        obs = np.array([
            normalized_price,
            balance_ratio,
            long_ratio,
            short_ratio,
            price_ma_ratio,
            volatility,
            volume_ma_ratio
        ], dtype=np.float32)
        
        # Clip observation values to prevent extreme values
        obs = np.clip(obs, -10, 10)
        
        return obs

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Long: {self.long_position}, Short: {self.short_position}")

    def get_portfolio_value(self):
        return self.balance + (self.long_position - self.short_position) * self.data[self.current_step]['close']