# nn.py
# Source: https://github.com/hojonathanho/diffusion/blob/master/
# diffusion_tf/nn.py
# Tensorflow 2.4.0
# Windows/MacOS/Linux
# Python 3.7


import math
import tensorflow as tf


def default_init(scale):
    return tf.initializers.variance_scaling(
        scale=1e-10 if scale == 0 else scale,
        mode="fan_avg",
        distribution="uniform")


def meanflat(x):
	return tf.math.reduce_mean(x, axis=list(range(1, len(x.shape))))


def get_timestep_embedding(timesteps, embedding_dim):
	# From fairseq. Build sinusoidal embeddings. This matches the 
	# implementation in tensor2tensor, but differs slightly from the
	# description in Section 3.5 of "Attention Is All You Need".
	assert len(timesteps.shape) == 1 # and timesteps.dtype == tf.int32

	half_dim = embedding_dim // 2
	emb = math.log(10000) / (half_dim - 1)
	emb = tf.math.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
	# emb = tf.range(num_embeddings, dtype=tf.float32)[:, None] * emb[None, :]
	emb = tf.cast(timesteps, dtype=tf.float32)[:, None] * emb[None, :]
	emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=1)
	if embedding_dim % 2 == 1: # zero pad.
		# emb = tf.concat([emb, tf.zeros([num_embeddings, 1])], axis=1)
		emb = tf.pad(emb, [[0, 0], [0, 1]])
	assert emb.shape == [timesteps.shape[0], embedding_dim]
	return emb