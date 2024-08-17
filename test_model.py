import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from environment import TradingEnvironment
from data_handler import DataHandler
from config import Config

def test_model(model, env):
    obs = env.reset()
    done = False
    portfolio_values = []
    actions_taken = []
    rewards = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        portfolio_values.append(env.get_portfolio_value())
        actions_taken.append(action)
        rewards.append(reward)

    return portfolio_values, actions_taken, rewards

def plot_results(portfolio_values, actions_taken, stock_prices):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

    # Portfolio value over time
    ax1.plot(portfolio_values)
    ax1.set_title('Portfolio Value Over Time')
    ax1.set_ylabel('Portfolio Value')

    # Actions taken over time
    ax2.plot([a[0] for a in actions_taken], label='Long')
    ax2.plot([a[1] for a in actions_taken], label='Short')
    ax2.set_title('Actions Taken Over Time')
    ax2.set_ylabel('Action Value')
    ax2.legend()

    # Stock price over time
    ax3.plot(stock_prices)
    ax3.set_title('Stock Price Over Time')
    ax3.set_ylabel('Price')
    ax3.set_xlabel('Time Steps')

    plt.tight_layout()
    plt.savefig('trading_results.png')
    plt.close()

def calculate_metrics(initial_balance, portfolio_values, rewards):
    total_return = (portfolio_values[-1] - initial_balance) / initial_balance * 100
    sharpe_ratio = np.mean(rewards) / np.std(rewards) if np.std(rewards) != 0 else 0
    max_drawdown = np.min([0] + [((np.max(portfolio_values[:i+1]) - v) / np.max(portfolio_values[:i+1])) * 100 for i, v in enumerate(portfolio_values)])
    
    return total_return, sharpe_ratio, max_drawdown

def main():
    config = Config()
    
    data_handler = DataHandler(config.stock_symbol, config.start_date, config.end_date)
    test_data = data_handler.get_test_data()
    env = TradingEnvironment(
        test_data, 
        config.initial_balance, 
        config.transaction_cost_pct, 
        config.borrow_cost_pct, 
        config.margin_requirement,
        config.max_loss_pct
    )

    # Load the trained model
    model = SAC.load("sac_trading_model")

    # Test the model
    portfolio_values, actions_taken, rewards = test_model(model, env)

    # Calculate metrics
    total_return, sharpe_ratio, max_drawdown = calculate_metrics(config.initial_balance, portfolio_values, rewards)

    # Print statistics
    print(f"Initial Balance: ${config.initial_balance:.2f}")
    print(f"Final Portfolio Value: ${portfolio_values[-1]:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2f}%")

    # Plot results
    plot_results(portfolio_values, actions_taken, [d['close'] for d in test_data])

if __name__ == "__main__":
    main()