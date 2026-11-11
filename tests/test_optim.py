"""Unit tests for numcompute_stream.optim."""
import numpy as np
import pytest

from numcompute_stream import optim


def quad(x):
    return float(np.sum(x ** 2))


def vec(x):
    return np.array([x[0] * x[1], np.sin(x[0]), x[1] ** 3])


def vec_jac(x):
    return np.array([[x[1], x[0]], [np.cos(x[0]), 0.0], [0.0, 3 * x[1] ** 2]])


def test_grad_central_and_forward_match_analytic():
    x = [1.0, -2.0, 3.0]
    exact = np.array([2.0, -4.0, 6.0])
    cen = optim.grad(quad, x)
    fwd = optim.grad(quad, x, method="forward")
    np.testing.assert_allclose(cen, exact, atol=1e-8)
    np.testing.assert_allclose(fwd, exact, atol=1e-3)
    assert np.abs(cen - exact).max() < np.abs(fwd - exact).max()   # central is more accurate


def test_grad_rosenbrock_and_single_variable():
    f = lambda p: 100 * (p[1] - p[0] ** 2) ** 2 + (1 - p[0]) ** 2
    exact = [-400 * 1.2 * (1.2 - 1.2 ** 2) + 0.4, 200 * (1.2 - 1.2 ** 2)]
    np.testing.assert_allclose(optim.grad(f, [1.2, 1.2]), exact, rtol=1e-5)
    np.testing.assert_allclose(optim.grad(lambda x: x[0] ** 3, [2.0]), [12.0], rtol=1e-6)


def test_grad_shape_integer_input_empty_and_constant_function():
    g = optim.grad(quad, np.arange(5))
    assert g.shape == (5,) and g.dtype == float
    assert optim.grad(lambda x: 1.0, []).shape == (0,)
    np.testing.assert_array_equal(optim.grad(lambda x: 5.0, [1.0, 2.0]), [0.0, 0.0])


def test_grad_does_not_modify_x_even_if_function_mutates_its_argument():
    x = np.array([1.0, 2.0])

    def mutating(p):
        s = float(np.sum(p ** 2))
        p[:] = 0.0
        return s

    g = optim.grad(mutating, x)
    np.testing.assert_array_equal(x, [1.0, 2.0])
    np.testing.assert_allclose(g, [2.0, 4.0], atol=1e-8)


def test_grad_non_contiguous_input_and_evaluation_counts():
    x = np.arange(10, dtype=float)[::3]
    assert not x.flags["C_CONTIGUOUS"]
    np.testing.assert_allclose(optim.grad(quad, x), 2 * x, atol=1e-8)
    calls = []
    f = lambda p: calls.append(1) or float(np.sum(p))
    optim.grad(f, np.zeros(4), method="central")
    assert len(calls) == 8                                         # 2n
    calls.clear()
    optim.grad(f, np.zeros(4), method="forward")
    assert len(calls) == 5                                         # n + 1


def test_grad_invalid_inputs():
    with pytest.raises(TypeError):
        optim.grad("not callable", [1.0])
    with pytest.raises(ValueError):
        optim.grad(quad, [[1.0, 2.0]])                             # 2-D x
    with pytest.raises(ValueError):
        optim.grad(quad, [1.0, np.nan])
    with pytest.raises(ValueError):
        optim.grad(quad, [1.0], method="backward")
    with pytest.raises(ValueError):
        optim.grad(quad, [1.0], h=-1e-5)
    with pytest.raises(TypeError):
        optim.grad(quad, [1.0], h="1e-5")


def test_grad_rejects_vector_output_and_non_finite_output():
    with pytest.raises(ValueError, match="scalar"):
        optim.grad(lambda x: np.array([1.0, 2.0]), [1.0])
    with np.errstate(all="ignore"), pytest.raises(ValueError, match="non-finite"):
        optim.grad(lambda x: np.log(x[0]), [0.0])


def test_jacobian_matches_analytic_both_methods():
    x = np.array([0.5, 1.5])
    J = optim.jacobian(vec, x)
    assert J.shape == (3, 2)
    np.testing.assert_allclose(J, vec_jac(x), atol=1e-8)
    np.testing.assert_allclose(optim.jacobian(vec, x, method="forward"), vec_jac(x), atol=1e-3)


def test_jacobian_scalar_function_linear_map_and_empty_x():
    J = optim.jacobian(quad, [1.0, 2.0, 3.0])
    assert J.shape == (1, 3)
    np.testing.assert_allclose(J[0], optim.grad(quad, [1.0, 2.0, 3.0]))
    A = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 4.0]])
    np.testing.assert_allclose(optim.jacobian(lambda x: A @ x, [0.3, 0.2, 0.1]), A, atol=1e-8)
    assert optim.jacobian(lambda x: np.array([1.0, 2.0]), []).shape == (2, 0)


def test_jacobian_invalid_inputs_and_inconsistent_output_length():
    with pytest.raises(TypeError):
        optim.jacobian(None, [1.0])
    with pytest.raises(ValueError):
        optim.jacobian(lambda x: np.ones((2, 2)), [1.0])           # matrix output
    with pytest.raises(ValueError, match="non-finite"):
        optim.jacobian(lambda x: np.array([1.0, np.nan]), [1.0])
    state = {"n": 0}

    def ragged(x):
        state["n"] += 1
        return np.ones(1 + state["n"] % 2)

    with pytest.raises(ValueError, match="same number"):
        optim.jacobian(ragged, [1.0, 2.0])


def test_relative_step_handles_large_x():
    f = lambda x: float(x[0] ** 2)
    x = [1e10]
    assert optim.grad(f, x, relative=True)[0] == pytest.approx(2e10, rel=1e-6)
    assert optim.grad(f, x)[0] != pytest.approx(2e10, rel=1e-6)   # fixed h loses accuracy here
    np.testing.assert_allclose(optim.grad(quad, [1.0, -2.0], relative=True), [2.0, -4.0], rtol=1e-6)
    np.testing.assert_allclose(optim.jacobian(vec, [1.0, 2.0], relative=True),
                               optim.jacobian(vec, [1.0, 2.0]), rtol=1e-6)
    with pytest.raises(TypeError):
        optim.grad(quad, [1.0], relative="yes")