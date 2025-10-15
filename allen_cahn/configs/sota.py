import ml_collections

import jax.numpy as jnp
# 在配置中添加时间戳参数
import datetime


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train"
    # config.mode = "eval"

    
    max_steps = 500000
    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-AllenCahn"
    config.wandb.name = f"no_causal_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # wandb.name = "no_causal"
    wandb.tag = None

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "Mlp"
    arch.num_layers = 4
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "tanh"
    arch.periodicity = ml_collections.ConfigDict(
        {"period": (jnp.pi,), "axis": (1,), "trainable": (False,)}
    )
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}
    )

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = max_steps
    training.batch_size_per_device = 64

    # Weighting
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"
    # weighting.init_weights = ml_collections.ConfigDict({"ics": 1.0, "res": 1.0})
    weighting.init_weights = ml_collections.ConfigDict({
        "ics": 1.0, 
        "r1": 1.0,
        "r2": 1.0,
        "r3": 1.0,
        "r4": 1.0,
        })
    weighting.momentum = 0.9
    weighting.update_every_steps = 1000

    weighting.use_causal = None
    weighting.causal_tol = 1.0
    weighting.num_chunks = 32

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_preds = False

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = max_steps
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models
    config.input_dim = 2

    # Integer for PRNG random seed.
    config.seed = 42

    return config
