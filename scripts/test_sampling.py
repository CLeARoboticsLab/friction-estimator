import jax.numpy as jp
import jax

joint_lims_max = jp.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)
joint_lims_min = jp.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)

x = jax.random.uniform(
    jax.random.key(1),
    (7,),
    minval=joint_lims_min,
    maxval=joint_lims_max,
)

print(x)