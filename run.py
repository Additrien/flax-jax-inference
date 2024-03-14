import functools
import timeit
from typing import Optional, Callable

import numpy as np

import jax
from jax import grad, jit, lax, random, numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax.lax import with_sharding_constraint
from jax.experimental import mesh_utils

import flax
from flax.training.common_utils import shard
from flax import struct, traverse_util, linen as nn
from flax.core import freeze, unfreeze
from flax.training import train_state, checkpoints
from transformers import AutoTokenizer, MixtralModel, FlaxMixtralModel,FlaxMixtralForCausalLM, MixtralConfig

import optax

def mesh_sharding(pspec: PartitionSpec) -> NamedSharding:
    return NamedSharding(mesh, pspec)   


def configure_model(model_path, tokenizer_path):
    # model_path = "hf-internal-testing/Mixtral-tiny"
    config = MixtralConfig.from_pretrained(model_path)
    config.max_position_embeddings = 128
    model = FlaxMixtralForCausalLM(config=config, dtype=jnp.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return model, tokenizer
    # prompt = "Hey, are you conscious? Can you talk to me?"
    # fx_inputs = tokenizer(prompt, return_tensors="jax")


def init_fn(k, input_ids, attention_mask, model, optimizer):
  variables = model.module.init(rngs=k, input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
  state = train_state.TrainState.create( # Create a `TrainState`.
    apply_fn=model.module.apply,
    params=variables['params'],
    tx=optimizer
  )
  return state


@functools.partial(jax.jit, in_shardings=(state_sharding, x_sharding), 
                   out_shardings=state_sharding)
def train_step(model, state, x):
# A fake loss function.
    def loss_unrolled(params):
        y = model.apply({'params': params}, x)
        return y.sum()
    grad_fn = jax.grad(loss_unrolled)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@functools.partial(jax.jit, in_shardings=(state_sharding, x_sharding), 
                   out_shardings=x_sharding)
def apply_fn(state, x):
    return state.apply_fn({'params': state.params}, x)


def block_all(x, y, xs):
    jax.tree_map(lambda x: x.block_until_ready(), lambda y: y.block_until_ready(),xs)
    return xs


def main(model_path):
   
    model, tokenizer = configure_model(model_path)

    device_mesh = mesh_utils.create_device_mesh((2, 4))
    mesh = Mesh(devices=device_mesh, axis_names=('data', 'model'))

    input_ids_sharding = mesh_sharding(PartitionSpec('data', None)) # dimensions: (batch, length)
    input_shape = (2, 1)
    input_ids = jnp.zeros(input_shape, dtype="i4")
    input_ids = jax.device_put(input_ids, input_ids_sharding)

    attention_mask_sharding = mesh_sharding(PartitionSpec('data', None)) # dimensions: (batch, length)
    attention_mask = jnp.ones_like(input_ids)
    attention_mask = jax.device_put(attention_mask, attention_mask_sharding)

    rng = jax.random.PRNGKey(0)
    optimizer = optax.adam(learning_rate=0.001)


    abstract_variables = jax.eval_shape(
        functools.partial(init_fn, model=model, optimizer=optimizer),
        k=rng,
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    # This `state_sharding` has the same pytree structure as `state`, the output
    # of the `init_fn`.
    state_sharding = nn.get_sharding(abstract_variables, mesh)

    jit_init_fn = jax.jit(init_fn, static_argnums=(3, 4),
                      in_shardings=(mesh_sharding(None), input_ids_sharding, attention_mask_sharding),
                      out_shardings=state_sharding)

    initialized_state = jit_init_fn(rng, input_ids, attention_mask, model, optimizer)

    with mesh:
        new_state = train_step(initialized_state, input_ids, attention_mask)

    with mesh:
        y = apply_fn(new_state, input_ids, attention_mask)
    print(type(y))
    print(y.dtype)
    print(y.shape)

    code_to_test = """
    with mesh:
        new_state = block_all(apply_fn(initialized_state, input_ids, attention_mask))
    """

    print(timeit.repeat(code_to_test, repeat=20))

