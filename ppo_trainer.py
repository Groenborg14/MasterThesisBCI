import torch.optim as optim
import torch
import torch.nn as nn

def compute_returns(rewards, values, gamma=0.99):
    returns = []
    G = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        G = r + gamma * G
        returns.insert(0, G)
    return torch.tensor(returns)

def ppo_train(agent, env, epochs=10, rollout_len=256, clip=0.2, gamma=0.99, lr=2.5e-4):
    optimizer = optim.Adam(agent.parameters(), lr=lr)

    for epoch in range(epochs):
        states, actions, log_probs, rewards, values = [], [], [], [], []
        state = env.reset()

        for _ in range(rollout_len):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob, entropy = agent.get_action(state_tensor)
            value = agent.forward(state_tensor)[1]

            next_state, reward, done, _ = env.step(action)

            # Store rollout
            states.append(state_tensor.squeeze(0))
            actions.append(torch.tensor(action))
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.squeeze(0))

            state = next_state
            if done:
                state = env.reset()

        returns = compute_returns(rewards, values, gamma)
        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(log_probs).detach()

        for _ in range(4):  # PPO mini-epochs
            logits, state_values = agent(states)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratios = (new_log_probs - old_log_probs).exp()
            advantages = returns - state_values.squeeze()

            # PPO loss
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - clip, 1 + clip) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            critic_loss = nn.MSELoss()(state_values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
