import torch
from torch import Tensor


def get_rewards(
    scores: Tensor, old_logp: Tensor, ref_logp: Tensor, mask: Tensor, kl_coef: float
) -> Tensor:
    # scores (B,)
    # old_logp, ref_logp (B, T)
    B, T = old_logp.shape
    rewards = -kl_coef * (old_logp - ref_logp)
    # last (B,)
    last = (torch.arange(T) * mask).argmax(-1)
    rewards[torch.arange(B), last] += scores
    return rewards


def GAE(rewards: Tensor, values: Tensor, mask: Tensor, gamma: float, lam: float):
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    for b in range(B):
        L = mask[b].sum()
        for i in reversed(range(L)):
            delta = (
                rewards[b, i]
                + gamma * (0 if i == L - 1 else values[b, i + 1])
                - values[b, i]
            )
            adv[b, i] = delta + gamma * lam * (0 if i == L - 1 else adv[b, i + 1])
    return adv, adv + values


def ppo_loss(
    scores: Tensor,
    values: Tensor,
    old_value: Tensor,
    logp: Tensor,
    old_logp: Tensor,
    ref_logp: Tensor,
    mask: Tensor,
    kl_coef: float,
    v_coef: float,
    gamma: float,
    lam: float,
    eps: float,
) -> Tensor:
    rewards = get_rewards(scores, old_logp, ref_logp, mask, kl_coef)
    adv, returns = GAE(rewards, old_value, mask, gamma, lam)
    ratio = torch.exp(logp - old_logp)
    policy_loss = -torch.minimum(ratio * adv, torch.clip(ratio, 1 - eps, 1 + eps) * adv)
    value_loss = (returns - values) ** 2
    return policy_loss + v_coef * value_loss
