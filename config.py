class Config:
    def __init__(self):
        # Environment settings
        self.stock_symbol = "AAPL"  # Example: Apple stock
        self.start_date = "2010-01-01"
        self.end_date = "2023-01-01"
        self.initial_balance = 10000
        self.transaction_cost_pct = 0.001  # 0.1% transaction cost
        self.borrow_cost_pct = 0.0001  # 0.01% borrow cost for short positions
        self.margin_requirement = 0.5  # 50% margin requirement for short positions
        self.max_loss_pct = 0.5  # 50% maximum loss before ending episode

        # SAC hyperparameters
        self.learning_rate = 0.0003  # Slightly increased learning rate
        self.buffer_size = 1000000  # Increased buffer size
        self.batch_size = 256
        self.gamma = 0.99
        self.tau = 0.005
        self.ent_coef = 'auto'
        self.train_freq = 1
        self.gradient_steps = 1

        # Training settings
        self.total_timesteps = 100000  # 1 million timesteps for full training
        self.eval_episodes = 10  # Number of episodes to run during evaluation

        # Model saving and logging
        self.save_freq = 100000  # Save model every 100k steps
        self.log_freq = 10000  # Log stats every 10k steps

    def update_learning_rate(self, new_lr):
        self.learning_rate = new_lr

    def update_total_timesteps(self, new_timesteps):
        self.total_timesteps = new_timesteps