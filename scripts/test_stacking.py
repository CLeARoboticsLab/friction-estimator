import jax
import jax.numpy as jnp
import jax.tree_util as jtu

example_tree = [1, 21, jnp.array([1,2,3])]
leaves = jtu.tree_leaves(example_tree)
print(f"{repr(example_tree):<45} has {len(leaves)} leaves: {leaves}")


def tree_stack(trees):
    return jtu.tree_map(lambda *v: jnp.stack(v), *trees)


new_tree = tree_stack([example_tree, example_tree])

leaves = jtu.tree_leaves(new_tree)
print(f"{repr(new_tree):<45} has {len(leaves)} leaves: {leaves}")