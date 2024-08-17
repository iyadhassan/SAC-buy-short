from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from environment import TradingEnvironment
from data_handler import DataHandler
from config import Config

class LearningRateScheduler(BaseCallback):
    def __init__(self, initial_lr=0.0003, min_lr=0.00001, decay_factor=0.5, decay_steps=250000, verbose=0):
        super(LearningRateScheduler, self).__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        progress = min(1.0, self.num_timesteps / self.decay_steps)
        new_lr = max(self.min_lr, self.initial_lr * (self.decay_factor ** progress))
        self.model.learning_rate = new_lr
        if self.verbose > 0:
            print(f"Timestep: {self.num_timesteps}, Learning Rate: {new_lr}")
        return True

def main():
    config = Config()
    data_handler = DataHandler(config.stock_symbol, config.start_date, config.end_date)
    
    train_data = data_handler.get_train_data()
    print(f"Number of training data points: {len(train_data)}")
    print(f"First few training data points: {train_data[:5]}")
    
    env = TradingEnvironment(
        train_data, 
        config.initial_balance, 
        config.transaction_cost_pct, 
        config.borrow_cost_pct, 
        config.margin_requirement,
        config.max_loss_pct
    )
    env = DummyVecEnv([lambda: env])

    model = SAC("MlpPolicy", env, verbose=1, 
                learning_rate=config.learning_rate,
                buffer_size=config.buffer_size,
                batch_size=config.batch_size,
                gamma=config.gamma,
                tau=config.tau,
                ent_coef=config.ent_coef,
                train_freq=config.train_freq,
                gradient_steps=config.gradient_steps)

    lr_scheduler = LearningRateScheduler(
        initial_lr=config.learning_rate,
        min_lr=config.learning_rate / 10,
        decay_factor=0.5,
        decay_steps=config.total_timesteps // 4,
        verbose=1
    )

    model.learn(total_timesteps=config.total_timesteps, callback=lr_scheduler)

    # Save the trained model
    model.save("sac_trading_model")

if __name__ == "__main__":
    main()