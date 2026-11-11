# mlrt
ML Runtime

This project tries to implement a distribtued ML runtime.

## Python environment

Dependencies are managed with [uv](https://docs.astral.sh/uv/).
The project uses Python 3.13 and PyTorch.

```sh
uv sync --locked
uv run python src/rl/grpo.py
```

`uv run` automatically uses the project-local `.venv`. For an interactive shell:

```sh
source .venv/bin/activate
```

Check PyTorch and Apple Silicon GPU availability:

```sh
uv run python -c "import torch; print(torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

Add dependencies with `uv add <package>` and keep `uv.lock` in version control.
PyTorch uses the default PyPI distribution, following the
[uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Unit tests

```sh
uv sync --locked
uv run pytest -q
```

Tests cover group advantages (sample standard deviation), the K3 KL estimator,
per-token broadcasting, PPO clipping for both advantage signs, and gradients.
