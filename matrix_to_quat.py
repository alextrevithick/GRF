import numpy as np
import tensorflow as tf

def safe_unsigned_div(a, b, eps=None, name=None):
  with tf.compat.v1.name_scope(name, 'safe_unsigned_div', [a, b, eps]):
    a = tf.convert_to_tensor(value=a)
    b = tf.convert_to_tensor(value=b)
    if eps is None:
      eps = 10.0 * np.finfo(np.float32).tiny
    eps = tf.convert_to_tensor(value=eps, dtype=b.dtype)

    return a / (b + eps)

def matrix2quat(rotation_matrix, name=None):
  with tf.compat.v1.name_scope(name, "quaternion_from_rotation_matrix",
                               [rotation_matrix]):
    rotation_matrix = tf.convert_to_tensor(value=rotation_matrix)

    trace = tf.linalg.trace(rotation_matrix)
    eps_addition = 2.0 * np.finfo(np.float32).eps
    rows = tf.unstack(rotation_matrix, axis=-2)
    entries = [tf.unstack(row, axis=-1) for row in rows]

    def tr_positive():
      sq = tf.sqrt(trace + 1.0) * 2.  # sq = 4 * qw.
      qw = 0.25 * sq
      qx = safe_unsigned_div(entries[2][1] - entries[1][2], sq)
      qy = safe_unsigned_div(entries[0][2] - entries[2][0], sq)
      qz = safe_unsigned_div(entries[1][0] - entries[0][1], sq)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_1():
      sq = tf.sqrt(1.0 + entries[0][0] - entries[1][1] - entries[2][2] +
                   eps_addition) * 2.  # sq = 4 * qx.
      qw = safe_unsigned_div(entries[2][1] - entries[1][2], sq)
      qx = 0.25 * sq
      qy = safe_unsigned_div(entries[0][1] + entries[1][0], sq)
      qz = safe_unsigned_div(entries[0][2] + entries[2][0], sq)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_2():
      sq = tf.sqrt(1.0 + entries[1][1] - entries[0][0] - entries[2][2] +
                   eps_addition) * 2.  # sq = 4 * qy.
      qw = safe_unsigned_div(entries[0][2] - entries[2][0], sq)
      qx = safe_unsigned_div(entries[0][1] + entries[1][0], sq)
      qy = 0.25 * sq
      qz = safe_unsigned_div(entries[1][2] + entries[2][1], sq)
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_3():
      sq = tf.sqrt(1.0 + entries[2][2] - entries[0][0] - entries[1][1] +
                   eps_addition) * 2.  # sq = 4 * qz.
      qw = safe_unsigned_div(entries[1][0] - entries[0][1], sq)
      qx = safe_unsigned_div(entries[0][2] + entries[2][0], sq)
      qy = safe_unsigned_div(entries[1][2] + entries[2][1], sq)
      qz = 0.25 * sq
      return tf.stack((qx, qy, qz, qw), axis=-1)

    def cond_idx(cond):
      cond = tf.expand_dims(cond, -1)
      cond = tf.tile(cond, [1] * (rotation_matrix.shape.ndims - 2) + [4])
      return cond

    where_2 = tf.compat.v1.where(
        cond_idx(entries[1][1] > entries[2][2]), cond_2(), cond_3())
    where_1 = tf.compat.v1.where(
        cond_idx((entries[0][0] > entries[1][1])
                 & (entries[0][0] > entries[2][2])), cond_1(), where_2)
    quat = tf.compat.v1.where(cond_idx(trace > 0), tr_positive(), where_1)
    return quat
