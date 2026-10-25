import torch
from torch import Tensor


# rewards: (B, G)
# advantage: (B, G)
def advantage(rewards: Tensor):
    return (rewards - rewards.mean(dim=-1, keepdim=True)) / (
        rewards.std(dim=-1, keepdim=True) + 1e-6
    )


def k3(logprob: Tensor, ref_logprob: Tensor):
    r = ref_logprob - logprob
    return torch.exp(r) - r - 1


def grpo(
    rewards: Tensor,
    logprob: Tensor,
    oldprod: Tensor,
    ref_logprob: Tensor,
    mask: Tensor,
    eps: float,
    beta: float,
):
    adv = advantage(rewards).unsqueeze(-1)
    # logprob, oldprob. r (B, G, T)
    r = torch.exp(logprob - oldprod)
    # surr (B, G, T)
    surr = torch.minimum(r * adv, torch.clip(r, 1 - eps, 1 + eps) * adv)
    loss = -surr + beta * k3(logprob, ref_logprob)
    # mask (B, G, T)
    # mask.sum(-1) (B, G)
    # loss.sum(-1) (B, G)
    return ((loss * mask).sum(-1) / mask.sum(-1)).mean(-1)


if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    rewards = torch.tensor([[0.0, 2.0]], device=device)
    old_logprob = torch.full((1, 2, 3), -2.0, device=device)
    logprob = torch.full_like(old_logprob, -1.95, requires_grad=True)
    mask = torch.ones_like(old_logprob, dtype=torch.bool)

    loss = grpo(rewards, logprob, old_logprob, old_logprob, mask, eps=0.2, beta=0.1)
    loss.mean().backward()
    print(f"GRPO device: {device}")
    print(f"GRPO loss: {loss.item():.6f}")
    print(f"Gradient norm: {logprob.grad.norm().item():.6f}")
