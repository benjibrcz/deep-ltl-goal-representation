from dataclasses import dataclass


@dataclass
class PPOConfig:
    """Configuration for PPO."""
    epochs: int = 4  # number of PPO epochs
    batch_size: int = 256  # batch size
    steps_per_process: int = 2048  # number of steps per process before update
    discount: float = 0.99  # discount factor
    lr: float = 0.0003  # learning rate
    gae_lambda: float = 0.95  # lambda coefficient in GAE formula
    entropy_coef: float = 0.01  # entropy term coefficient
    value_loss_coef: float = 0.5  # value loss term coefficient
    max_grad_norm: float = 0.5  # maximum norm of gradient
    optim_eps: float = 1e-8  # Adam optimizer epsilon
    clip_eps: float = 0.2  # clipping epsilon for PPO
