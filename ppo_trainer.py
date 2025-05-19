# ppo_trainer.py
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_returns(rewards, gamma=0.99):
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)


def ppo_train(agent, env, epochs=500, rollout_len=1024, clip=0.2, gamma=0.99, lr=5e-5):
    optimizer = optim.Adam(agent.parameters(), lr=lr)
    agent.train()

    for epoch in range(epochs):
        states, actions, log_probs, rewards, entropies, values = [], [], [], [], [], []
        state = env.reset()
        episode_rewards = 0
        episode_steps = 0

        for step in range(rollout_len):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action, log_prob, entropy, confidence = agent.get_action(state_tensor)
            _, value = agent(state_tensor)

            # === Pass (action, confidence) tuple to env ===
            next_state, reward, done, info = env.step((action, confidence))

            episode_rewards += reward
            episode_steps += 1

            # === Store rollout ===
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            values.append(value.squeeze(0).detach().cpu().numpy())

            state = next_state
            if done:
                break
        #print("Value estimates:", values[:5])
        # === Prepare tensors ===
        returns = compute_returns(rewards, gamma).to(device)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions).to(device)
        log_probs_old = torch.stack(log_probs).detach().to(device)
        values = torch.tensor(np.array(values), dtype=torch.float32).to(device)
        advantages = returns - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # === PPO Update ===
        for _ in range(5):
            logits, new_values = agent(states.unsqueeze(1))
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            new_values = new_values.squeeze(1)

            ratio = (new_log_probs - log_probs_old).exp()
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            #critic_loss = nn.MSELoss()(new_values, returns)
            value_pred_clipped = values + (new_values - values).clamp(-0.2, 0.2)
            value_losses = (new_values - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            loss = actor_loss + 0.5 * critic_loss# - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
            optimizer.step()

        # === Logging ===
        wandb.log({
            "train/epoch": epoch + 1,
            "train/loss": loss.item(),
            "train/reward_episode": episode_rewards / episode_steps if episode_steps > 0 else 0,
            "train/actor_loss": actor_loss.item(),
            "train/critic_loss": critic_loss.item(),
            "train/entropy": entropy.item(),
        })
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} | Episode Reward: {episode_rewards:.2f}")
