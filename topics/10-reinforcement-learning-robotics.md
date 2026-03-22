# Reinforcement Learning for Robotics and Autonomous Driving

## Introduction

Reinforcement Learning (RL) enables robots and autonomous vehicles to learn from experience. This document covers RL algorithms, applications in autonomous systems, and best practices for training RL agents in simulation and real-world environments.

## RL Fundamentals

### Markov Decision Process

```python
class MDP:
    """Markov Decision Process formulation"""
    
    def __init__(self, states, actions, transitions, rewards, gamma=0.99):
        self.states = states          # State space S
        self.actions = actions        # Action space A
        self.transitions = transitions  # P(s'|s,a)
        self.rewards = rewards        # R(s,a,s')
        self.gamma = gamma            # Discount factor
        
    def step(self, state, action):
        """Take a step in the MDP"""
        # Sample next state
        next_state = self.sample_next_state(state, action)
        
        # Get reward
        reward = self.rewards(state, action, next_state)
        
        # Check if terminal
        done = self.is_terminal(next_state)
        
        return next_state, reward, done
    
    def get_transition_prob(self, state, action, next_state):
        """Get transition probability P(s'|s,a)"""
        return self.transitions.get((state, action, next_state), 0)
    
    def sample_next_state(self, state, action):
        """Sample next state from transition distribution"""
        probs = []
        next_states = []
        
        for s_prime in self.states:
            prob = self.get_transition_prob(state, action, s_prime)
            if prob > 0:
                probs.append(prob)
                next_states.append(s_prime)
        
        return np.random.choice(next_states, p=probs)
```

### RL Agent Interface

```python
class RLAgent:
    """Base RL agent interface"""
    
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
    def select_action(self, state, explore=True):
        """Select action given state"""
        raise NotImplementedError
    
    def update(self, experiences):
        """Update policy from experiences"""
        raise NotImplementedError
    
    def save(self, path):
        """Save agent"""
        raise NotImplementedError
    
    def load(self, path):
        """Load agent"""
        raise NotImplementedError
```

## Value-Based Methods

### Deep Q-Network (DQN)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

class DQN(nn.Module):
    """Deep Q-Network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        return self.net(state)


class DQNAgent(RLAgent):
    """DQN agent with experience replay"""
    
    def __init__(self, state_dim, action_dim, 
                 lr=1e-3, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=64):
        super().__init__(state_dim, action_dim)
        
        # Networks
        self.q_network = DQN(state_dim, action_dim)
        self.target_network = DQN(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Experience replay
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Update counter
        self.update_count = 0
        self.target_update_freq = 1000
        
    def select_action(self, state, explore=True):
        """Epsilon-greedy action selection"""
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def update(self, experiences=None):
        """Update Q-network"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample batch
        if experiences is None:
            batch = random.sample(self.replay_buffer, self.batch_size)
        else:
            batch = experiences
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(dim=1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
```

### Double DQN

```python
class DoubleDQNAgent(DQNAgent):
    """Double DQN to reduce overestimation"""
    
    def update(self, experiences=None):
        """Update with Double DQN"""
        if len(self.replay_buffer) < self.batch_size:
            return 0
        
        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Current Q values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use online network for action selection
        with torch.no_grad():
            # Select actions using online network
            next_actions = self.q_network(next_states).argmax(dim=1)
            
            # Evaluate actions using target network
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze()
            
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Loss
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
```

### Dueling DQN

```python
class DuelingDQN(nn.Module):
    """Dueling network architecture"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Shared feature extraction
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        features = self.feature(state)
        
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values
```

## Policy Gradient Methods

### REINFORCE

```python
class REINFORCEAgent(RLAgent):
    """REINFORCE policy gradient"""
    
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99):
        super().__init__(state_dim, action_dim)
        
        # Policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        
        # Episode storage
        self.log_probs = []
        self.rewards = []
        
    def select_action(self, state, explore=True):
        """Sample action from policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy(state_tensor)
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        
        # Store log probability
        self.log_probs.append(dist.log_prob(action))
        
        return action.item()
    
    def store_reward(self, reward):
        """Store reward"""
        self.rewards.append(reward)
    
    def update(self, experiences=None):
        """Update policy using collected trajectory"""
        # Compute returns
        returns = []
        R = 0
        
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.FloatTensor(returns)
        
        # Normalize
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute loss
        log_probs = torch.stack(self.log_probs)
        loss = -(log_probs * returns).sum()
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.log_probs = []
        self.rewards = []
        
        return loss.item()
```

### Actor-Critic

```python
class ActorCritic(nn.Module):
    """Actor-Critic network"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state):
        shared = self.shared(state)
        
        action_logits = self.actor(shared)
        value = self.critic(shared)
        
        return action_logits, value


class A2CAgent(RLAgent):
    """Advantage Actor-Critic"""
    
    def __init__(self, state_dim, action_dim, lr=7e-4, gamma=0.99, 
                 value_loss_coef=0.5, entropy_coef=0.01):
        super().__init__(state_dim, action_dim)
        
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        
    def select_action(self, state, explore=True):
        """Select action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            logits, value = self.network(state_tensor)
            probs = F.softmax(logits, dim=-1)
            
            if explore:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            else:
                action = probs.argmax(dim=-1)
        
        return action.item()
    
    def update(self, experiences):
        """Update from batch of experiences"""
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)
        
        # Forward pass
        logits, values = self.network(states)
        _, next_values = self.network(next_states)
        
        # Compute advantages
        returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = returns - values.squeeze()
        
        # Policy loss
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1))
        policy_loss = -(action_log_probs.squeeze() * advantages.detach()).mean()
        
        # Value loss
        value_loss = F.mse_loss(values.squeeze(), returns.detach())
        
        # Entropy bonus
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + \
               self.value_loss_coef * value_loss - \
               self.entropy_coef * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
        self.optimizer.step()
        
        return loss.item()
```

### PPO (Proximal Policy Optimization)

```python
class PPOAgent(RLAgent):
    """Proximal Policy Optimization"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_epsilon=0.2, 
                 value_loss_coef=0.5, entropy_coef=0.01,
                 num_epochs=10, batch_size=64):
        super().__init__(state_dim, action_dim)
        
        self.network = ActorCritic(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        
    def compute_gae(self, rewards, values, next_values, dones):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = next_values
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self, trajectory):
        """Update policy with PPO"""
        states = torch.FloatTensor(trajectory['states'])
        actions = torch.LongTensor(trajectory['actions'])
        rewards = trajectory['rewards']
        dones = trajectory['dones']
        
        # Get old probabilities
        with torch.no_grad():
            logits, values = self.network(states)
            old_log_probs = F.log_softmax(logits, dim=-1).gather(
                1, actions.unsqueeze(1)
            )
            old_values = values.squeeze()
        
        # Compute advantages
        next_value = self.network(
            torch.FloatTensor(trajectory['last_state'])
        )[1].item()
        
        advantages = self.compute_gae(rewards, old_values.numpy(), next_value, dones)
        returns = advantages + old_values.detach()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.num_epochs):
            # Mini-batch update
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                mb_indices = indices[start:end]
                
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                
                # Forward pass
                logits, values = self.network(mb_states)
                log_probs = F.log_softmax(logits, dim=-1).gather(
                    1, mb_actions.unsqueeze(1)
                )
                
                # Ratio
                ratio = torch.exp(log_probs - mb_old_log_probs)
                
                # Clipped objective
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.clip_epsilon, 
                    1 + self.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values.squeeze(), mb_returns)
                
                # Entropy
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
                
                # Total loss
                loss = policy_loss + \
                       self.value_loss_coef * value_loss - \
                       self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
        
        return loss.item()
```

## Continuous Action Spaces

### DDPG (Deep Deterministic Policy Gradient)

```python
class DDPGAgent(RLAgent):
    """Deep Deterministic Policy Gradient for continuous actions"""
    
    def __init__(self, state_dim, action_dim, max_action,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005):
        super().__init__(state_dim, action_dim)
        
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        
        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(1000000)
        
    def select_action(self, state, explore=True, noise_scale=0.1):
        """Select continuous action"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if explore:
            # Add exploration noise
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -self.max_action, self.max_action)
        
        return action
    
    def update(self, experiences=None):
        """Update actor and critic"""
        if len(self.replay_buffer) < 64:
            return 0, 0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(64)
        
        # Compute target Q
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # Critic update
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor update
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update targets
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        return actor_loss.item(), critic_loss.item()
    
    def soft_update(self, target, source):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )


class Actor(nn.Module):
    """Actor network for DDPG"""
    
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )
        
        self.max_action = max_action
        
    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Critic network for DDPG"""
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )
        
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
```

## Application: Robot Locomotion

### Sim-to-Real Transfer

```python
class SimToRealTransfer:
    """Transfer policies from simulation to real robot"""
    
    def __init__(self, sim_env, real_robot):
        self.sim_env = sim_env
        self.real_robot = real_robot
        self.domain_randomization = DomainRandomization()
        
    def train_in_sim(self, num_episodes=10000):
        """Train policy in simulation with domain randomization"""
        agent = PPOAgent(
            state_dim=self.sim_env.observation_space.shape[0],
            action_dim=self.sim_env.action_space.shape[0]
        )
        
        for episode in range(num_episodes):
            # Randomize dynamics
            self.domain_randomization.apply(self.sim_env)
            
            state = self.sim_env.reset()
            trajectory = {'states': [], 'actions': [], 'rewards': [], 'dones': []}
            
            done = False
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, info = self.sim_env.step(action)
                
                trajectory['states'].append(state)
                trajectory['actions'].append(action)
                trajectory['rewards'].append(reward)
                trajectory['dones'].append(done)
                
                state = next_state
            
            trajectory['last_state'] = state
            agent.update(trajectory)
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Reward: {sum(trajectory['rewards'])}")
        
        return agent
    
    def transfer_to_real(self, agent):
        """Transfer to real robot with fine-tuning"""
        # Deploy on real robot
        state = self.real_robot.get_state()
        
        for step in range(1000):
            action = agent.select_action(state, explore=False)
            self.real_robot.execute_action(action)
            
            # Collect real data
            next_state = self.real_robot.get_state()
            reward = self.real_robot.compute_reward()
            
            # Fine-tune with small learning rate
            agent.update([{
                'state': state,
                'action': action,
                'reward': reward,
                'next_state': next_state,
                'done': False
            }])
            
            state = next_state


class DomainRandomization:
    """Randomize simulation parameters"""
    
    def __init__(self):
        self.randomization_params = {
            'mass': (0.8, 1.2),
            'friction': (0.5, 1.5),
            'latency': (0, 50),  # ms
            'noise': (0.0, 0.1),
            'gravity': (9.5, 10.0)
        }
        
    def apply(self, env):
        """Apply randomization to environment"""
        # Randomize physics
        env.mass = np.random.uniform(*self.randomization_params['mass'])
        env.friction = np.random.uniform(*self.randomization_params['friction'])
        env.gravity = np.random.uniform(*self.randomization_params['gravity'])
        
        # Randomize sensor noise
        env.observation_noise = np.random.uniform(*self.randomization_params['noise'])
        
        # Randomize actuator latency
        env.actuator_latency = np.random.uniform(*self.randomization_params['latency'])
```

## Best Practices

### 1. Reward Shaping

```python
def shaped_reward(state, action, next_state, goal):
    """Reward function with shaping for faster learning"""
    reward = 0
    
    # Goal reward
    distance_to_goal = np.linalg.norm(next_state[:2] - goal)
    reward += 10.0 / (distance_to_goal + 1.0)
    
    # Progress reward
    prev_distance = np.linalg.norm(state[:2] - goal)
    reward += (prev_distance - distance_to_goal) * 5.0
    
    # Action penalty (smoothness)
    reward -= 0.1 * np.sum(action ** 2)
    
    # Collision penalty
    if check_collision(next_state):
        reward -= 100.0
    
    # Goal reached bonus
    if distance_to_goal < 0.1:
        reward += 100.0
    
    return reward
```

### 2. Curriculum Learning

```python
class CurriculumLearning:
    """Progressively increase task difficulty"""
    
    def __init__(self, env):
        self.env = env
        self.difficulty = 0.0
        
    def update_difficulty(self, success_rate):
        """Adjust difficulty based on performance"""
        if success_rate > 0.8:
            self.difficulty = min(1.0, self.difficulty + 0.1)
        elif success_rate < 0.3:
            self.difficulty = max(0.0, self.difficulty - 0.1)
    
    def configure_env(self):
        """Configure environment based on difficulty"""
        # Easier: fewer obstacles, larger goal region
        self.env.num_obstacles = int(5 + 15 * self.difficulty)
        self.env.goal_radius = 0.5 - 0.4 * self.difficulty
        self.env.max_episode_length = int(500 - 300 * self.difficulty)
```

### 3. Safe Exploration

```python
class SafeExplorer:
    """Ensure safe exploration during training"""
    
    def __init__(self, safety_constraints):
        self.constraints = safety_constraints
        
    def safe_action(self, state, proposed_action):
        """Modify action to satisfy safety constraints"""
        # Check constraints
        for constraint in self.constraints:
            if not constraint.is_satisfied(state, proposed_action):
                # Project to safe action
                proposed_action = constraint.project_to_safe(state, proposed_action)
        
        return proposed_action
    
    def is_state_safe(self, state):
        """Check if state is safe"""
        for constraint in self.constraints:
            if not constraint.is_state_safe(state):
                return False
        return True
```

## Conclusion

Reinforcement learning provides powerful tools for training autonomous systems. From value-based methods like DQN to policy gradient methods like PPO, each approach has strengths for different problems. Key to success is proper reward design, simulation-to-real transfer, and safe exploration strategies.

## References

- "Reinforcement Learning: An Introduction" (Sutton & Barto)
- "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- "Continuous Control with Deep RL" (Lillicrap et al., 2015)
- Tesla AI Day presentations

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Word Count**: ~4000 words  
**Size**: ~23KB
