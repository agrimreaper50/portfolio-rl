import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import gym
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt

# Custom Trading Environment for Multiple Assets with Simplified Action Space
class TradingEnv(gym.Env):
    def __init__(self, data, assets, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.assets = assets  # List of asset symbols
        self.initial_balance = initial_balance
        self.num_assets = len(self.assets)
        # Action space: Separate action for each asset (0: Hold, 1: Buy, 2: Sell)
        self.action_space = spaces.MultiDiscrete([3] * self.num_assets)
        # Observation space: Prices for each asset + holdings + balance + net worth
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(self.num_assets * 2 + 2,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.holdings = np.zeros(self.num_assets, dtype=np.float32)
        self.net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.trades = []
        # History for visualization
        self.history = {'net_worth': [], 'balance': [], 'holdings': [], 'prices': [], 'actions': []}
        return self._next_observation()

    def _next_observation(self):
        # Get the current prices for all assets
        prices = self.data.iloc[self.current_step][[asset + '_Close' for asset in self.assets]].values.astype(np.float32)
        obs = np.concatenate([
            prices,
            self.holdings,
            np.array([self.balance, self.net_worth], dtype=np.float32)
        ])
        return obs

    def step(self, actions):
        prices = self.data.iloc[self.current_step][[asset + '_Close' for asset in self.assets]].values.astype(np.float32)
        self.current_step += 1

        # Transaction cost percentage
        transaction_cost_pct = 0.001  # 0.1% per trade

        for i, action in enumerate(actions):
            current_price = prices[i]
            # Skip if price is invalid
            if np.isnan(current_price) or current_price <= 0:
                continue  # Skip actions for this asset

            if action == 1:  # Buy
                # Calculate how many shares we can buy
                max_buy = self.balance // (current_price * (1 + transaction_cost_pct))
                if max_buy > 0:
                    # Buy as many as possible
                    shares_to_buy = max_buy
                    cost = shares_to_buy * current_price * (1 + transaction_cost_pct)
                    self.balance -= cost
                    self.holdings[i] += shares_to_buy
                    self.trades.append({'step': self.current_step, 'asset': self.assets[i], 'type': 'buy', 'shares': shares_to_buy, 'price': current_price})
            elif action == 2:  # Sell
                # Sell all holdings of this asset
                shares_to_sell = self.holdings[i]
                if shares_to_sell > 0:
                    revenue = shares_to_sell * current_price * (1 - transaction_cost_pct)
                    self.balance += revenue
                    self.holdings[i] -= shares_to_sell
                    self.trades.append({'step': self.current_step, 'asset': self.assets[i], 'type': 'sell', 'shares': shares_to_sell, 'price': current_price})
            # else: Hold (do nothing)

        prev_net_worth = self.net_worth
        # Ensure holdings and prices are valid
        valid_holdings = np.nan_to_num(self.holdings)
        valid_prices = np.nan_to_num(prices)
        self.net_worth = self.balance + np.sum(valid_holdings * valid_prices)
        self.max_net_worth = max(self.max_net_worth, self.net_worth)

        # Calculate reward as percentage change in net worth
        reward = (self.net_worth - prev_net_worth) / self.initial_balance

        # Done if we have depleted our balance or at the end of data
        done = self.net_worth <= self.initial_balance * 0.1 or self.current_step >= len(self.data) - 1

        # Record history
        self.history['net_worth'].append(self.net_worth)
        self.history['balance'].append(self.balance)
        self.history['holdings'].append(self.holdings.copy())
        self.history['prices'].append(prices)
        self.history['actions'].append(actions.copy())

        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        # Implement rendering if needed
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, Balance: {self.balance:.2f}, Holdings: {self.holdings}")

# Enhanced Neural Network with Additional Layers
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(p=0.2)
        self.fc4 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        return x  # Output is raw scores for each action combination

# DQN Agent with Modified Epsilon Decay and Learning Rate
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.95, lr=0.0001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size  # For MultiDiscrete action space
        self.gamma = gamma  # Discount factor
        self.lr = lr  # Learning rate
        self.batch_size = batch_size
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.95  # Adjusted decay rate
        self.epsilon_min = 0.01
        self.num_actions = 3 ** self.action_size.shape[0]
        self.model = DQN(state_size, self.num_actions).float()
        self.target_model = DQN(state_size, self.num_actions).float()
        self.update_target_model()
        # Weight decay added for regularization
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        # Early stopping parameters
        self.best_val_loss = np.inf
        self.early_stop_count = 0
        self.early_stop_patience = 15  # Increased patience
        self.validation_losses = []

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        # Flatten action for storage
        action_flat = self._flatten_action(action)
        self.memory.append((state, action_flat, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            # Random action
            return np.random.randint(0, 3, size=self.action_size.shape[0])
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        # Get action with highest Q-value
        action_idx = torch.argmax(q_values).item()
        action = self._unflatten_action(action_idx)
        return action

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.tensor(np.array([s for s, _, _, _, _ in minibatch]), dtype=torch.float32)
        actions = torch.tensor([a for _, a, _, _, _ in minibatch], dtype=torch.int64)
        rewards = torch.tensor([r for _, _, r, _, _ in minibatch], dtype=torch.float32)
        next_states = torch.tensor(np.array([s for _, _, _, s, _ in minibatch]), dtype=torch.float32)
        dones = torch.tensor([d for _, _, _, _, d in minibatch], dtype=torch.float32)

        # Compute target Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]
        targets = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = self.criterion(q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def validate(self, env):
        state = env.reset()
        total_loss = 0
        done = False
        while not done:
            action = self.act(state)
            next_state, reward, done, _ = env.step(action)
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_flat = self._flatten_action(action)
            action_tensor = torch.tensor([action_flat], dtype=torch.int64)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            done_tensor = torch.tensor([done], dtype=torch.float32)

            # Compute Q-values
            q_value = self.model(state_tensor).gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q_value = self.target_model(next_state_tensor).max(1)[0]
            target = reward_tensor + (1 - done_tensor) * self.gamma * next_q_value

            # Compute loss
            loss = self.criterion(q_value, target)
            total_loss += loss.item()
            state = next_state

        average_loss = total_loss / env.current_step
        self.validation_losses.append(average_loss)
        return average_loss

    def _flatten_action(self, action):
        # Convert MultiDiscrete action to single integer
        action_idx = 0
        for i, a in enumerate(action):
            action_idx += a * (3 ** i)
        return action_idx

    def _unflatten_action(self, action_idx):
        # Convert single integer back to MultiDiscrete action
        action = []
        for _ in range(self.action_size.shape[0]):
            action.append(action_idx % 3)
            action_idx //= 3
        return np.array(action[::-1])

# Data preprocessing function
def split_and_prepare_data(data, train_ratio=0.7, val_ratio=0.15, columns_to_normalize=None):
    data['Date'] = pd.to_datetime(data['Date'])
    # Determine split indices
    train_split = int(train_ratio * len(data))
    val_split = int((train_ratio + val_ratio) * len(data))
    # Normalize specified columns based on training data
    if columns_to_normalize:
        for col in columns_to_normalize:
            train_min = data.iloc[:train_split][col].min()
            train_max = data.iloc[:train_split][col].max()
            if train_max != train_min:
                data[col] = (data[col] - train_min) / (train_max - train_min)
            else:
                print(f"Warning: Column {col} has zero variance in training data. Skipping normalization.")
    # Split the data
    train_data = data.iloc[:train_split]
    val_data = data.iloc[train_split:val_split]
    test_data = data.iloc[val_split:]
    return train_data, val_data, test_data

# Load and prepare data
file_path = 'portfolio_data.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)
data = data.dropna()
assets = ['MSFT', 'AAPL', 'JPM']  # List of assets to include

# Include SPY_Close in columns if it's in your data
if 'SPY_Close' in data.columns:
    spy_data_available = True
else:
    spy_data_available = False
    print("SPY_Close column not found in data. SPY performance will not be plotted.")

# Only normalize volumes
columns_to_normalize = []
for asset in assets:
    columns_to_normalize.append(f'{asset}_Volume')

train_data, val_data, test_data = split_and_prepare_data(
    data,
    train_ratio=0.7,
    val_ratio=0.15,
    columns_to_normalize=columns_to_normalize
)

# Initialize environments and agent
train_env = TradingEnv(train_data, assets=assets)
val_env = TradingEnv(val_data, assets=assets)
test_env = TradingEnv(test_data, assets=assets)

state_size = train_env.observation_space.shape[0]
action_size = train_env.action_space
agent = DQNAgent(state_size=state_size, action_size=action_size)

# Training the DQN Agent with adjusted epsilon decay
episodes = 50  # Adjust as needed
for e in range(episodes):
    state = train_env.reset()
    total_reward = 0
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = train_env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        # Replay experience
        agent.replay()
    agent.update_target_model()

    # Decay epsilon after each episode
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay

    # Validate after each episode
    val_loss = agent.validate(val_env)
    print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward * train_env.initial_balance:.2f}, Validation Loss: {val_loss:.6f}, Epsilon: {agent.epsilon:.4f}")

    # Early stopping check
    if val_loss < agent.best_val_loss:
        agent.best_val_loss = val_loss
        agent.early_stop_count = 0
        # Optionally save the best model
        best_model_state = agent.model.state_dict()
    else:
        agent.early_stop_count += 1
    if agent.early_stop_count >= agent.early_stop_patience:
        print("Early stopping triggered.")
        # Restore the best model weights
        agent.model.load_state_dict(best_model_state)
        break

# Test the agent's performance
state = test_env.reset()
total_reward = 0
done = False

while not done:
    action = agent.act(state)  # Use the trained agent
    state, reward, done, _ = test_env.step(action)
    total_reward += reward

# Evaluation Metrics
def calculate_metrics(net_worths, initial_balance, risk_free_rate=0.01):
    net_worths_array = np.array(net_worths)
    # Daily Returns
    daily_returns = np.diff(net_worths_array) / net_worths_array[:-1]
    # Adjust for potential division by zero
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    # Average Daily Return
    average_daily_return = np.mean(daily_returns)
    # Standard Deviation of Daily Returns
    std_daily_return = np.std(daily_returns)
    # Annualize the Sharpe Ratio
    sharpe_ratio = ((average_daily_return - risk_free_rate / 252) / std_daily_return) * np.sqrt(252) if std_daily_return != 0 else 0.0
    # Total Return
    total_return = (net_worths_array[-1] - initial_balance) / initial_balance * 100
    # Maximum Drawdown
    peaks = np.maximum.accumulate(net_worths_array)
    drawdowns = (peaks - net_worths_array) / peaks
    max_drawdown = np.max(drawdowns)
    return total_return, sharpe_ratio, max_drawdown

# Calculate evaluation metrics for the agent
initial_balance = test_env.initial_balance
agent_total_return, agent_sharpe_ratio, agent_max_drawdown = calculate_metrics(
    test_env.history['net_worth'], initial_balance
)

# Baseline Strategy: Buy-and-Hold
def baseline_strategy(data, initial_balance=10000):
    initial_prices = data.iloc[0][[asset + '_Close' for asset in assets]].values.astype(np.float32)
    final_prices = data.iloc[-1][[asset + '_Close' for asset in assets]].values.astype(np.float32)
    # Allocate equal funds to each asset
    allocation = initial_balance / len(assets)
    shares_held = allocation / initial_prices
    final_balance = np.sum(shares_held * final_prices)
    return final_balance

# Calculate Baseline Net Worth Over Time
baseline_net_worths = []
allocation = initial_balance / len(assets)
shares_held = allocation / test_data.iloc[0][[asset + '_Close' for asset in assets]].values.astype(np.float32)
for idx in range(len(test_data)):
    prices = test_data.iloc[idx][[asset + '_Close' for asset in assets]].values.astype(np.float32)
    net_worth = np.sum(shares_held * prices)
    baseline_net_worths.append(net_worth)

# Calculate SPY Performance Over Time
if spy_data_available:
    spy_prices = test_data['SPY_Close'].values
    spy_initial_price = spy_prices[0]
    spy_net_worths = (spy_prices / spy_initial_price) * initial_balance

    # Calculate SPY Metrics
    spy_total_return, spy_sharpe_ratio, spy_max_drawdown = calculate_metrics(
        spy_net_worths, initial_balance
    )

# Plot portfolio vs Baseline and SPY performance
plt.figure(figsize=(12, 6))
plt.plot(test_env.history['net_worth'], label='DQN Agent Portfolio', color='green')
plt.plot(baseline_net_worths, label='Baseline (Buy-and-Hold)', linestyle='--', color='orange')
if spy_data_available:
    plt.plot(spy_net_worths, label='S&P 500 (SPY)', linestyle='-.', color='blue')
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Performance Comparison: DQN Agent vs Baseline vs S&P 500')
plt.legend()
plt.grid(True)
plt.show()

# Visualize Positions Over Time
actions_history = np.array(test_env.history['actions'])
holdings_history = np.array(test_env.history['holdings'])
prices_history = np.array(test_env.history['prices'])
time_steps = range(len(holdings_history))

for i, asset in enumerate(assets):
    plt.figure(figsize=(12, 4))
    asset_prices = [price[i] for price in prices_history]
    plt.plot(asset_prices, label=f'{asset} Price')
    buy_steps = [idx for idx, action in enumerate(actions_history) if action[i] == 1]
    sell_steps = [idx for idx, action in enumerate(actions_history) if action[i] == 2]
    plt.scatter(buy_steps, [asset_prices[idx] for idx in buy_steps], marker='^', color='g', label='Buy')
    plt.scatter(sell_steps, [asset_prices[idx] for idx in sell_steps], marker='v', color='r', label='Sell')
    plt.title(f'{asset} Trading Actions Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

# Print Metrics
print(f"Agent Performance on Test Data:")
print(f"  Total Reward: {total_reward * initial_balance:.2f}")
print(f"  Total Return: {agent_total_return:.2f}%")
print(f"  Sharpe Ratio: {agent_sharpe_ratio:.2f}")
print(f"  Maximum Drawdown: {agent_max_drawdown:.2f}")

# Print Baseline Metrics
baseline_total_return = (baseline_net_worths[-1] - initial_balance) / initial_balance * 100
print(f"Baseline Buy-and-Hold Performance:")
print(f"  Final Balance: {baseline_net_worths[-1]:.2f}")
print(f"  Total Return: {baseline_total_return:.2f}%")

if spy_data_available:
    print(f"S&P 500 Performance:")
    print(f"  Final Balance: {spy_net_worths[-1]:.2f}")
    print(f"  Total Return: {spy_total_return:.2f}%")
    print(f"  Sharpe Ratio: {spy_sharpe_ratio:.2f}")
    print(f"  Maximum Drawdown: {spy_max_drawdown:.2f}")
