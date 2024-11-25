import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import numpy as np
import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TradingEnv(gym.Env):
    def __init__(self, data, initial_balance=10000):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(57,), dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        return self._next_observation()

    def _next_observation(self):
        frame = self.data.iloc[self.current_step].values[1:]
        frame = frame.astype(float)  # Ensure numeric type
        return np.append(frame, [self.balance, self.shares_held, self.net_worth])[:57]

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['MSFT_Close']
        self.current_step += 1

        if action == 1 and self.balance >= current_price:
            self.shares_held += 1
            self.balance -= current_price

        elif action == 2 and self.shares_held > 0:
            self.shares_held -= 1
            self.balance += current_price

        self.net_worth = self.balance + self.shares_held * current_price
        reward = (self.net_worth - self.prev_net_worth) - 0.01 * abs(action - 1)
        self.prev_net_worth = self.net_worth
        done = self.current_step >= len(self.data) - 1
        return self._next_observation(), reward, done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Net Worth: {self.net_worth}")

# Neural network for the DQN agent
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, gamma=0.99, lr=0.001, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = DQN(state_size, action_size).float()
        self.target_model = DQN(state_size, action_size).float()
        self.update_target_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_state = np.array(next_state, dtype=np.float32)  # Convert to numeric array
                # print(f"State shape for NN: {state.shape}")  # Debug state shape
                next_state = torch.tensor(next_state).unsqueeze(0)
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            target_f = self.model(state).detach().numpy()
            target_f[0][action] = target
            self.optimizer.zero_grad()
            output = self.model(state)
            loss = nn.MSELoss()(output, torch.tensor(target_f))
            loss.backward()
            self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def split_and_prepare_data(data, train_ratio=0.8, val_ratio=0.1, columns_to_normalize=None):
    """
    Splits and prepares financial data into training, validation, and testing sets.
    Normalizes the specified columns using the training data's min and max values.
    
    Parameters:
        data (pd.DataFrame): The full dataset including a 'Date' column and price columns.
        train_ratio (float): Proportion of the data to use for training.
        val_ratio (float): Proportion of the data to use for validation.
        columns_to_normalize (list): List of column names to normalize.
        
    Returns:
        train_data (pd.DataFrame): Training dataset.
        val_data (pd.DataFrame): Validation dataset.
        test_data (pd.DataFrame): Testing dataset.
    """
    # Ensure the data contains a valid 'Date' column
    data['Date'] = pd.to_datetime(data['Date'])  # Keep this to ensure 'Date' is a datetime type
    
    # Determine split indices
    train_split = int(train_ratio * len(data))
    val_split = int((train_ratio + val_ratio) * len(data))
    
    # Split the data
    train_data = data.iloc[:train_split]
    val_data = data.iloc[train_split:val_split]
    test_data = data.iloc[val_split:]
    
    # Normalize the specified columns based on the training data
    if columns_to_normalize:
        for col in columns_to_normalize:
            train_min = train_data[col].min()
            train_max = train_data[col].max()
            
            # Apply normalization to the entire dataset
            data[col] = (data[col] - train_min) / (train_max - train_min)
    
    # Re-split the normalized data
    train_data = data.iloc[:train_split]
    val_data = data.iloc[train_split:val_split]
    test_data = data.iloc[val_split:]
    
    return train_data, val_data, test_data


# Load the CSV file
file_path = 'portfolio_data.csv'
data = pd.read_csv(file_path)
data = data.dropna()  # Remove rows with missing values

# Use the method to split and prepare the data
train_data, val_data, test_data = split_and_prepare_data(
    data, 
    train_ratio=0.8, 
    val_ratio=0.1, 
    columns_to_normalize=['MSFT_Close', 'SPY_Close']
)

# Display split sizes
len(train_data), len(val_data), len(test_data)


# Initialize environment and agent
env = TradingEnv(train_data)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size=state_size, action_size=action_size)

# Training the DQN Agent
episodes = 100
for e in range(episodes):
    state = env.reset()
    total_reward = 0
    for time in range(len(train_data) - 1):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.replay()
    agent.update_target_model()
    print(f"Episode: {e + 1}/{episodes}, Total Reward: {total_reward}")

# Evaluation Metrics
def calculate_metrics(net_worths, initial_balance, risk_free_rate=0.01):
    # Total Return
    total_return = (net_worths[-1] - initial_balance) / initial_balance * 100
    
    # Daily Returns
    daily_returns = np.diff(net_worths) / net_worths[:-1]
    
    # Sharpe Ratio
    sharpe_ratio = (np.mean(daily_returns) - risk_free_rate) / np.std(daily_returns)
    
    # Maximum Drawdown
    peaks = np.maximum.accumulate(net_worths)
    drawdowns = (peaks - net_worths) / peaks
    max_drawdown = np.max(drawdowns)
    
    return total_return, sharpe_ratio, max_drawdown

# Baseline Strategy: Buy-and-Hold
def baseline_strategy(data, initial_balance=10000):
    initial_price = data.iloc[0]['MSFT_Close']
    final_price = data.iloc[-1]['MSFT_Close']
    
    # Assume all money is invested at the beginning
    shares_held = initial_balance / initial_price
    final_balance = shares_held * final_price
    
    return final_balance

# Calculate S&P 500 (SPY) Performance
def calculate_spy_performance(data, initial_balance):
    spy_prices = data['SPY_Close'].values
    spy_normalized = (spy_prices / spy_prices[0]) * initial_balance  # Normalize to start at the same value
    return spy_normalized

# Test the agent's performance
test_env = TradingEnv(test_data)
state = test_env.reset()
total_reward = 0
done = False

net_worths = []
while not done:
    action = agent.act(state)  # Use the trained agent
    state, reward, done, _ = test_env.step(action)
    total_reward += reward
    net_worths.append(test_env.net_worth)

# Calculate S&P 500 normalized performance
spy_performance = calculate_spy_performance(test_data, test_env.initial_balance)

# Calculate evaluation metrics for the agent
initial_balance = test_env.initial_balance
total_return, sharpe_ratio, max_drawdown = calculate_metrics(net_worths, initial_balance)

print(data['SPY_Close'].describe())

# Print metrics
print(f"Agent Performance on Test Data:")
print(f"  Total Reward: {total_reward}")
print(f"  Total Return: {total_return:.2f}%")
print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
print(f"  Maximum Drawdown: {max_drawdown:.2f}")

# Calculate S&P 500 Total Return
spy_total_return = (spy_performance[-1] - spy_performance[0]) / spy_performance[0] * 100
print(f"S&P 500 Total Return: {spy_total_return:.2f}%")

# Plot portfolio vs S&P 500 performance
plt.figure(figsize=(12, 6))
plt.plot(net_worths, label='Agent Portfolio')
plt.plot(spy_performance, label='S&P 500 (SPY)', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Portfolio Value')
plt.title('Portfolio Performance vs S&P 500 Index')
plt.legend()
plt.show()