from jax import numpy as jp


def rot_tra_x(x: jp.ndarray, angle: float) -> jp.ndarray:
    """Rotates a vector x around the x-axis by angle."""
    c, s = jp.cos(angle), jp.sin(angle)
    T = jp.array([[1, 0, 0, 0], [0, c, -s, -s], [0, s, c, c], [0, 0, 0, 1]])
    return jp.round(jp.matmul(T, x), 2)


print(rot_tra_x(jp.array([0, 0, 0, 1]), 0))  # Should be [0, 0, 1, 1]
print(rot_tra_x(jp.array([0, 0, 0, 1]), jp.pi / 2))  # Should be [0, -1, 0, 1]
print(rot_tra_x(jp.array([0, 0, 0, 1]), jp.pi))  # Should be [0, 0, -1, 1]
print(rot_tra_x(jp.array([0, 0, 0, 1]), jp.pi*(1 + 1/2)))  # Should be [0, 1, 0, 1]


def TCoM1(q: float) -> jp.ndarray:
    """Returns the transformation matrix for the center of mass of link 1."""
    return jp.round(
        jp.array(
            [
                0,
                -1 / 2 * jp.sin(q[0] + q[1]) - jp.sin(q[0]),
                1 / 2 * jp.cos(q[0] + q[1]) + jp.cos(q[0]),
                1,
            ]
        ),
        2,
    )


print("\n")
print(TCoM1(jp.array([0, 0])))  # Should be [0, 0, 1.5, 1]
print(TCoM1(jp.array([jp.pi/2, 0])))  # Should be [0, -1.5, 0.0, 1]
print(TCoM1(jp.array([jp.pi, jp.pi/2])))  # Should be [0, 0.5, -1.0, 1]
