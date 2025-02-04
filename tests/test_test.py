"""This module contains dummies for the test module.
It will we be updated with real tests in the near future."""


def add(a, b):
    """Add two numbers together."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


def test_add():
    """Test the add function."""
    assert add(2, 3) == 5  # Test with positive numbers
    assert add(-1, 1) == 0  # Test with a negative number
    assert add(0, 0) == 0  # Test with zero


def test_subtract():
    """Test the subtract function."""
    assert subtract(5, 3) == 2  # Test with positive numbers
    assert subtract(3, 5) == -2  # Test with negative result
    assert subtract(0, 0) == 0  # Test with zero
