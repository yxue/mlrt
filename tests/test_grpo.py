import math

import pytest
import torch
from torch.testing import assert_close

from src.rl.grpo import advantage, grpo, k3


def test_advantage_normalizes_each_group_independently():
    rewards = torch.tensor([[1., 2., 3.], [10., 12., 14.]], dtype=torch.float64)
    # Sample standard deviations are 1 and 2; retain the implementation's epsilon.
    expected = torch.tensor([
        [-1 / (1 + 1e-6), 0., 1 / (1 + 1e-6)],
        [-2 / (2 + 1e-6), 0., 2 / (2 + 1e-6)],
    ], dtype=torch.float64)
    assert_close(advantage(rewards), expected)


def test_advantage_equal_rewards_is_finite_zero():
    actual = advantage(torch.tensor([[4., 4., 4.], [-2., -2., -2.]]))
    assert torch.isfinite(actual).all()
    assert_close(actual, torch.zeros_like(actual))


def test_k3_equal_policies_has_zero_value_and_gradient():
    logprob = torch.tensor([-3., -1., -.2], requires_grad=True)
    actual = k3(logprob, logprob.detach())
    assert_close(actual, torch.zeros_like(actual))
    actual.sum().backward()
    assert_close(logprob.grad, torch.zeros_like(logprob))


def test_k3_known_ratios_and_gradient():
    logprob = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    ref = torch.tensor([math.log(.5), 0., math.log(2.)], dtype=torch.float64)
    actual = k3(logprob, ref)
    expected = torch.tensor([.5 + math.log(2.) - 1, 0., 2 - math.log(2.) - 1], dtype=torch.float64)
    assert_close(actual, expected)
    assert (actual >= 0).all()
    actual.sum().backward()
    assert_close(logprob.grad, torch.tensor([.5, 0., -1.], dtype=torch.float64))


@pytest.mark.parametrize('shape', [(2, 3, 4), (3, 3, 3), (1, 3, 1)])
def test_grpo_broadcasts_group_advantages_over_tokens(shape):
    batch, group, _ = shape
    rewards = torch.tensor([[1., 2., 3.]], dtype=torch.float64).repeat(batch, 1)
    logprob = torch.zeros(shape, dtype=torch.float64)
    actual = grpo(rewards, logprob, logprob, logprob, mask=torch.ones_like(logprob), eps=.2, beta=.1)
    expected = torch.tensor([1 / (1 + 1e-6), 0., -1 / (1 + 1e-6)], dtype=torch.float64)
    assert actual.shape == (batch,)
    assert_close(actual, expected.mean().expand(batch))


@pytest.mark.parametrize('ratio', [.5, .9, 1., 1.1, 1.5])
def test_grpo_clipping_value_and_gradient_for_both_advantage_signs(ratio):
    rewards = torch.tensor([[0., 2.]], dtype=torch.float64)
    logprob = torch.full((1, 2, 3), -3. + math.log(ratio), dtype=torch.float64, requires_grad=True)
    old = torch.full_like(logprob, -3.)
    actual = grpo(rewards, logprob, old, logprob.detach(), mask=torch.ones_like(logprob), eps=.2, beta=0.)
    magnitude = 1 / (math.sqrt(2) + 1e-6)
    # Negative advantages clip low ratios; positive advantages clip high ratios.
    expected = torch.tensor([max(ratio, .8) * magnitude, -min(ratio, 1.2) * magnitude], dtype=torch.float64)
    assert_close(actual, expected.mean().reshape(1))
    actual.sum().backward()
    expected_grad = torch.tensor([
        ratio * magnitude if ratio > .8 else 0.,
        -ratio * magnitude if ratio < 1.2 else 0.,
    ], dtype=torch.float64)
    assert_close(logprob.grad, expected_grad.reshape(1, 2, 1).expand_as(logprob) / 6)


def test_grpo_equal_rewards_leaves_only_weighted_kl():
    rewards = torch.ones(2, 3, dtype=torch.float64)
    logprob = torch.full((2, 3, 4), -2., dtype=torch.float64, requires_grad=True)
    ref = torch.full_like(logprob, -2. + math.log(2.))
    actual = grpo(rewards, logprob, logprob.detach(), ref, mask=torch.ones_like(logprob), eps=.2, beta=.25)
    assert_close(actual, torch.full_like(actual, .25 * (1 - math.log(2.))))
    actual.sum().backward()
    assert_close(logprob.grad, torch.full_like(logprob, -.25 / 12))


@pytest.mark.parametrize('mask_dtype', [torch.bool, torch.float64])
def test_grpo_variable_lengths_keep_batch_and_zero_padding_gradients(mask_dtype):
    rewards = torch.tensor([[0., 2.], [2., 0.]], dtype=torch.float64)
    ratios = torch.tensor([
        [[.9, 1.1, 1.], [1.1, 1., .9]],
        [[.9, 1., 1.1], [1.1, .9, 1.]],
    ], dtype=torch.float64)
    mask = torch.tensor([
        [[1, 0, 0], [1, 1, 0]],
        [[1, 1, 1], [1, 0, 0]],
    ], dtype=mask_dtype)
    old = torch.full_like(ratios, -3.)
    logprob = (old + ratios.log()).requires_grad_()
    actual = grpo(rewards, logprob, old, logprob.detach(), mask, .2, 0.)
    a = 1 / (math.sqrt(2) + 1e-6)
    # Each rollout has equal weight, regardless of its valid token count.
    expected = torch.tensor([(.9 - 1.05) * a / 2, (-1. + 1.1) * a / 2], dtype=torch.float64)
    assert actual.shape == (2,)
    assert_close(actual, expected)
    actual.sum().backward()
    expected_grad = torch.tensor([
        [[.9 * a / 2, 0., 0.], [-1.1 * a / 4, -a / 4, 0.]],
        [[-.9 * a / 6, -a / 6, -1.1 * a / 6], [1.1 * a / 2, 0., 0.]],
    ], dtype=torch.float64)
    assert_close(logprob.grad, expected_grad)


def test_grpo_adding_finite_padding_preserves_loss_and_gradients():
    rewards = torch.tensor([[0., 2.]], dtype=torch.float64)
    logprob = torch.tensor([[[-2., -2.1], [-2.2, -2.3]]], dtype=torch.float64, requires_grad=True)
    old = torch.full_like(logprob, -2.2)
    ref = torch.full_like(logprob, -2.4)
    actual = grpo(rewards, logprob, old, ref, torch.ones_like(logprob), .2, .1)
    actual.sum().backward()

    def pad(tensor, value):
        return torch.cat([tensor.detach(), torch.full((1, 2, 3), value, dtype=tensor.dtype)], dim=-1)

    padded_logprob = pad(logprob, -8.).requires_grad_()
    mask = pad(torch.ones_like(logprob), 0.)
    padded = grpo(rewards, padded_logprob, pad(old, -5.), pad(ref, -3.), mask, .2, .1)
    assert_close(padded, actual)
    padded.sum().backward()
    assert_close(padded_logprob.grad[..., :2], logprob.grad)
    assert_close(padded_logprob.grad[..., 2:], torch.zeros(1, 2, 3, dtype=torch.float64))
