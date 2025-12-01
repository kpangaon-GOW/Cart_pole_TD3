import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dm_control import suite
import matplotlib.pyplot as plt
import os
from datetime import datetime
import imageio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.done = np.zeros((max_size, 1))

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[ind],
            self.action[ind],
            self.next_state[ind],
            self.reward[ind],
            self.done[ind]
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = self.q1(sa)
        q2 = self.q2(sa)
        return q1, q2
    
    def q1_forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa)


class TD3:
    def __init__(self, state_dim, action_dim, max_action, 
                 actor_lr=3e-4, critic_lr=3e-4, device='cpu'):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        
        self.tau = 0.005
        self.gamma = 0.99
        self.policy_delay = 2
        self.noise_clip = 0.5
        self.policy_noise = 0.2
        self.total_it = 0

    
    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if noise != 0.0:
            action = action + np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)
        
        return action
    
    def train(self, replay_buffer, batch_size):
        self.total_it += 1
        
        states, actions, next_states, rewards, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * self.policy_noise
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -self.max_action, self.max_action)
            
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            q_target = rewards + (1 - dones) * self.gamma * q_target
        
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self._update_targets()
    
    def _update_targets(self):
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


def build_plots(cumulative_rewards_per_seed, episode_rewards_per_seed, seeds):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for seed, cum_rewards in zip(seeds, cumulative_rewards_per_seed):
        axes[0].plot(cum_rewards, label=f'Seed {seed}', alpha=0.7)
    axes[0].set_xlabel('Timestep')
    axes[0].set_ylabel('Cumulative Reward')
    axes[0].set_title('Cumulative Rewards per Timestep')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for seed, ep_rewards in zip(seeds, episode_rewards_per_seed):
        axes[1].plot(ep_rewards, label=f'Seed {seed}', alpha=0.7)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Episode Reward')
    axes[1].set_title('Episode Rewards')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved training plots: training_plots_{timestamp}.png")
    plt.show()


def plot_average_rewards_with_std(episode_rewards_per_seed, seeds):
    max_len = max(len(rewards) for rewards in episode_rewards_per_seed)
    padded_rewards = np.array([
        np.pad(rewards, (0, max_len - len(rewards)), mode='edge') 
        for rewards in episode_rewards_per_seed
    ])
    
    mean_rewards = np.mean(padded_rewards, axis=0)
    std_rewards = np.std(padded_rewards, axis=0)
    episodes = np.arange(1, len(mean_rewards) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, mean_rewards, color='green', linewidth=2, label='Mean Reward')
    plt.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards, 
                     color='green', alpha=0.3, label='± 1 Std Dev')
    
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title(f'Average Episode Rewards Across Seeds {seeds}\n(Shaded region = ± 1 Standard Deviation)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'average_rewards_with_std_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved average rewards plot: average_rewards_with_std_{timestamp}.png")
    plt.show()
    
    print(f"\nSummary Statistics Across Seeds {seeds}:")
    print(f"  Final Mean Reward: {mean_rewards[-1]:.4f} ± {std_rewards[-1]:.4f}")
    print(f"  Overall Mean Reward: {np.mean(mean_rewards):.4f} ± {np.mean(std_rewards):.4f}")


def save_episode_video(frames, filename):
    os.makedirs('videos', exist_ok=True)
    video_path = os.path.join('videos', filename)
    imageio.mimsave(video_path, frames, fps=30)
    print(f"Saved video: {video_path}")


def train_agent(seed, num_episodes=50, max_steps=1000):
    print(f"Training with seed {seed}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = suite.load(domain_name='cartpole', task_name='balance', task_kwargs={"random": seed})
    
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    
    action_dim = action_spec.shape[0]
    state_dim = sum(obs.shape[0] for obs in observation_spec.values())
    max_action = float(action_spec.maximum[0])
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    print(f"Max action: {max_action}")
    print(f"Using device: {device}\n")
    
    agent = TD3(state_dim, action_dim, max_action, device=device)
    replay_buffer = ReplayBuffer(state_dim, action_dim)
    
    episode_rewards = []
    cumulative_rewards = []
    cumulative_sum = 0.0
    
    for episode in range(num_episodes):
        time_step = env.reset()
        state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])
        episode_reward = 0.0
        
        for step in range(max_steps):
            action = agent.select_action(state, noise=0.1 * max_action)
            time_step = env.step(action)
            next_state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])
            reward = time_step.reward
            done = time_step.last()
            
            replay_buffer.add(state, action, next_state, reward, done)
            
            episode_reward += reward
            cumulative_sum += reward
            cumulative_rewards.append(cumulative_sum)
            
            state = next_state
            
            if replay_buffer.size > 64:
                agent.train(replay_buffer, batch_size=64)
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        print(f"Seed {seed} | Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.4f} | Cumulative: {cumulative_sum:.4f}")
    
    env.close()
    actor_path = f'/home/k/EEE_598_HW_cartpole/td3_cartpole_actor_seed_{seed}.pt'
    critic_path = f'/home/k/EEE_598_HW_cartpole/td3_cartpole_critic_seed_{seed}.pt'
    torch.save(agent.actor.state_dict(), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)
    print(f"Model saved for seed {seed}!")
    print(f"  Actor: {actor_path}")
    print(f"  Critic: {critic_path}")
    
    return episode_rewards, cumulative_rewards, agent


def evaluate_all_checkpoints(training_seeds, eval_seed=10, max_steps=10000):
    print(f"Evaluating all checkpoints on seed {eval_seed}")    
    np.random.seed(eval_seed)
    torch.manual_seed(eval_seed)
    
    env = suite.load(domain_name='cartpole', task_name='balance', task_kwargs={"random": eval_seed})
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    action_dim = action_spec.shape[0]
    state_dim = sum(obs.shape[0] for obs in observation_spec.values())
    max_action = float(action_spec.maximum[0])
    env.close()
    
    all_timestep_rewards = []
    all_frames = []  # Collect frames for video
    
    for train_seed in training_seeds:
        print(f"\nEvaluating checkpoint from training seed {train_seed}...")
        
        agent = TD3(state_dim, action_dim, max_action, device=device)
        actor_path = f'/home/k/EEE_598_HW_cartpole/td3_cartpole_actor_seed_{train_seed}.pt'
        agent.actor.load_state_dict(torch.load(actor_path, map_location=device))
        agent.actor.eval()
        
        np.random.seed(eval_seed)
        torch.manual_seed(eval_seed)
        env = suite.load(domain_name='cartpole', task_name='balance', task_kwargs={"random": eval_seed})
        
        time_step = env.reset()
        state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])
        
        timestep_rewards = []
        frames = []
        for step in range(max_steps):
            # Capture frame for video
            frame = env.physics.render(height=480, width=640)
            frames.append(frame)
            
            action = agent.select_action(state, noise=0.0)
            time_step = env.step(action)
            
            next_state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])
            reward = time_step.reward
            timestep_rewards.append(reward)
            
            state = next_state
            
            if time_step.last():
                break
        
        all_timestep_rewards.append(timestep_rewards)
        all_frames.append(frames)
        total_reward = sum(timestep_rewards)
        print(f"  Checkpoint seed {train_seed}: Total reward = {total_reward:.4f}, Steps = {len(timestep_rewards)}")
        
        # Save video for this checkpoint
        if frames:
            video_filename = f'eval_seed_{eval_seed}_checkpoint_{train_seed}_reward_{total_reward:.4f}.mp4'
            save_episode_video(frames, video_filename)
        
        env.close()
    
    max_len = max(len(rewards) for rewards in all_timestep_rewards)
    padded_rewards = np.array([
        np.pad(rewards, (0, max_len - len(rewards)), mode='constant', constant_values=0)
        for rewards in all_timestep_rewards
    ])
    
    mean_rewards = np.mean(padded_rewards, axis=0)
    std_rewards = np.std(padded_rewards, axis=0)
    timesteps = np.arange(1, len(mean_rewards) + 1)
    
 
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, mean_rewards, color='blue', linewidth=2, label='Mean Reward')
    plt.fill_between(timesteps, mean_rewards - std_rewards, mean_rewards + std_rewards,
                     color='blue', alpha=0.3, label='± 1 Std Dev')
    
    plt.xlabel('Timestep')
    plt.ylabel('Reward')
    plt.title(f'Evaluation on Seed {eval_seed}: Reward vs Timestep\n(Across checkpoints from training seeds {training_seeds})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'evaluation_reward_vs_timestep_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved evaluation plot: evaluation_reward_vs_timestep_{timestamp}.png")
    plt.show()
    
    print(f"\nEvaluation Summary (Seed {eval_seed}):")
    print(f"  Mean Total Reward: {np.sum(mean_rewards):.4f}")
    print(f"  Std of Total Rewards: {np.std([sum(r) for r in all_timestep_rewards]):.4f}")
    print(f"  Mean Reward per Timestep: {np.mean(mean_rewards):.4f} ± {np.mean(std_rewards):.4f}")
    
    return all_timestep_rewards, mean_rewards, std_rewards


def evaluate_agent(agent, seed=10, num_episodes=5, max_steps=1000):
    print(f"Evaluating agent with seed {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    env = suite.load(domain_name='cartpole', task_name='balance', task_kwargs={"random": seed})
    
    action_spec = env.action_spec()
    observation_spec = env.observation_spec()
    
    action_dim = action_spec.shape[0]
    state_dim = sum(obs.shape[0] for obs in observation_spec.values())
    max_action = float(action_spec.maximum[0])
    
    eval_rewards = []
    all_frames = []
    
    for episode in range(num_episodes):
        time_step = env.reset()
        state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])
        episode_reward = 0.0
        
        for step in range(max_steps):
            frame = env.physics.render(height=480, width=640)
            all_frames.append(frame)
            
            action = agent.select_action(state, noise=0.0)
            time_step = env.step(action)
            
            next_state = np.concatenate([obs.flatten() for obs in time_step.observation.values()])
            reward = time_step.reward
            episode_reward += reward
            
            state = next_state
            
            if time_step.last():
                break
        
        eval_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes} | Reward: {episode_reward:.4f}")
    
    if all_frames:
        eval_video_filename = f'evaluation_seed_{seed}_avg_reward_{np.mean(eval_rewards):.4f}.mp4'
        save_episode_video(all_frames, eval_video_filename)
    
    env.close()
    return eval_rewards


if __name__ == "__main__":
    #seeds = [0]
    seeds = [0, 1, 2]
    all_episode_rewards = []
    all_cumulative_rewards = []
    trained_agents = {}
    
    for seed in seeds:
        episode_rewards, cumulative_rewards, agent = train_agent(
            seed=seed,
            num_episodes=100,
            max_steps=1000
        )
        all_episode_rewards.append(episode_rewards)
        all_cumulative_rewards.append(cumulative_rewards)
        trained_agents[seed] = agent
    
    build_plots(all_cumulative_rewards, all_episode_rewards, seeds)
    plot_average_rewards_with_std(all_episode_rewards, seeds)
    print("Evaluation of the checkpoints")
    evaluate_all_checkpoints(training_seeds=seeds, eval_seed=10, max_steps=1000)
