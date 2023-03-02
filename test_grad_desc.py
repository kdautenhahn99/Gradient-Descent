'''This file contains unit tests for gradient_descent.py.'''
import pytest
from gradient_descent import grad_desc, stoch_grad_desc

loss, slope, intrcpt = grad_desc()
loss2, slope2, intrcpt2 = stoch_grad_desc()


# Test Valid Values
def test_valid_input():
    # Check for gradient end cases
    assert round(loss[-1]) == 0
    assert round(slope[-1]) == 4
    assert round(intrcpt[-1]) == 2
    # Check for stochastic end cases
    assert round(loss2[-1]) == 0
    assert round(slope2[-1]) == 4
    assert round(intrcpt2[-1]) == 2
    # Check for equality in gradient and stochastic ends
    assert round(loss[-1]) == round(loss2[-1])
    assert round(slope[-1]) == round(slope2[-1])
    assert round(intrcpt[-1]) == round(intrcpt2[-1])
    # Check that beginning value is bigger/smaller than end value
    assert round(loss[0]) > round(loss[-1])
    assert round(loss2[0]) > round(loss2[-1])
    assert round(slope[0]) < round(slope[-1])
    assert round(slope2[0]) < round(slope2[-1])
    assert round(intrcpt[0]) < round(intrcpt[-1])
    assert round(intrcpt2[0]) < round(intrcpt2[-1])


# Test Invalid Values
def test_invalid_input():
    with pytest.raises(ValueError):
        grad_desc(learn_rate=-5)
        grad_desc(learn_rate=0)
