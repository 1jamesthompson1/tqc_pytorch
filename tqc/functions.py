import torch

from tqc import DEVICE


def eval_policy(policy, eval_env, max_episode_steps, eval_episodes=10):
    policy.eval()
    rewards = []
    episode_steps = []
    for _ in range(eval_episodes):
        state, done = eval_env.reset(), False
        state = state[0]
        t = 0
        ep_reward = 0.
        while not done and t < max_episode_steps:
            action = policy.select_action(state)
            state, reward, terminated, truncated,  _ = eval_env.step(action)
            done = terminated or truncated
            ep_reward += reward
            t += 1
        
        rewards.append(ep_reward)
        episode_steps.append(t)
    policy.train()
    return rewards, episode_steps


def quantile_huber_loss_f(quantiles, samples, kappa = 1):
    pairwise_delta = samples[:, None, None, :] - quantiles[:, :, :, None]  # batch x nets x quantiles x samples
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > kappa,
                             abs_pairwise_delta - 0.5,
                             pairwise_delta ** 2 * 0.5)

    n_quantiles = quantiles.shape[2]
    tau = torch.arange(n_quantiles, device=DEVICE).float() / n_quantiles + 1 / 2 / n_quantiles
    loss = (torch.abs(tau[None, None, :, None] - (pairwise_delta < 0).float()) * huber_loss).mean()
    return loss
