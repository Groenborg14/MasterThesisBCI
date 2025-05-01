import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np
import wandb    

def compute_returns(rewards, values, gamma=0.99):
    # Compute the discounted returns for the rewards
    # PPO need discounted returns to estimate how good the action was
    # this loops backward through the rewards list and applies the discount factor gamma
    returns = []
    G = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)

def ppo_train(agent, env, epochs=400, rollout_len=256, clip=0.2, gamma=0.97, lr=2e-4):
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    for epoch in range(epochs):
        # Collect rollout (a batch of transitions)
        states, actions, log_probs, rewards, entropies, values = [], [], [], [], [], []
        state = env.reset()

        for _ in range(rollout_len):
            # Convert EEG vector to pytorch tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            # Sample action from the agent
            action, log_prob, entropy = agent.get_action(state_tensor)
            # Get value estimate from the current state
            value = agent.forward(state_tensor)[1]
            # take a step in the env using the sampled action
            next_state, reward, done, _ = env.step(action)

            # Store rollout
            states.append(state_tensor.squeeze(0))
            actions.append(torch.tensor(action))
            log_probs.append(log_prob)
            rewards.append(reward)
            entropies.append(entropy)
            values.append(value.squeeze(0))

            state = next_state
            if done:
                state = env.reset()

        returns = compute_returns(rewards, values, gamma)
        states = torch.stack(states)
        actions = torch.stack(actions)
        log_probs_old = torch.stack(log_probs).detach()
        values = torch.stack(values)
        #advantages = returns - values.detach()

        for _ in range(6):  # PPO mini-epochs
            # PPO uses multiple epochs of mini-batch updates for better learning
            # common for PPO to collect a batch of transitions (rollout) and the use that batch to update the policy and value function
            # typically between 3-10 epochs of mini-batch updates
            # Recomputes log prob of old actions under the new policy
            logits, new_values = agent(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Convert logits to probabilities for logging
            probs = torch.softmax(logits, dim=-1)
            avg_probs = probs.mean(dim=0)  # average over the batch for a clean wandb plot

            # Log to wandb
            wandb.log({
                "avg_prob_class_0": avg_probs[0].item(),
                "avg_prob_class_1": avg_probs[1].item(),
                "logtis max": logits.max().item(),
                "logits min": logits.min().item()
            })
            wandb.log({
                "prob_dist": wandb.Histogram(probs.detach().cpu().numpy())
            })

            # Calculates the ratio between new and old policy for policy change
            ratio = (new_log_probs - log_probs_old).exp()
            # Measures how much better or worse the action was than expected
            # Returns: The cumulative rewards from the current state to the end of the episode
            # values: The value estimates from the current state
            advantages = returns - new_values.squeeze()

            # PPO loss
            # PPO clipped objective function
            # Avoids making huge policy updates by clipping the ratio between new and old policy
            # surrogate1 is the original objective function
            # surrogate2 is the clipped objective function
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1 - clip, 1 + clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            # Critic loss ensure value estimates are accurate
            # Entropy loss encourages exploration
            # The total loss is a weighted combination of the actor loss, critic loss, and entropy loss
            critic_loss = nn.MSELoss()(new_values.squeeze(), returns)
            # Entropy encourages exploration by penalizing certainty in the policy
            # The entropy term is multiplied by a weight that decays over time to reduce exploration as training progresses
            #initial_entropy_weight = 0.001
            #entropy_weight = initial_entropy_weight *(0.95 ** epoch)  # Decay entropy weight over time
            loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

            # Standard PyTorch training step
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)

            optimizer.step()
            
        # 🧠 Log CSP projections for visualization
        if epoch == 0 or epoch == epochs - 1:
            trial_flat = states[0].detach().cpu().numpy()
            wandb.log({
                f"csp_features_epoch{epoch+1}": wandb.Histogram(trial_flat)
            })

        # 📊 Log training metrics
        wandb.log({
            "epoch": epoch + 1,
            "loss": loss.item(),
            "reward_avg": np.mean(rewards),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy.item()
        })
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f} | Reward avg: {np.mean(rewards):.2f}")
