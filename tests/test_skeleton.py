# -*- coding: utf-8 -*-

import pytest
from myhelpers.skeleton import fib

__author__ = "Mohannad Elhamod"
__copyright__ = "Mohannad Elhamod"
__license__ = "mit"


def test_fib():
    assert fib(1) == 1
    assert fib(2) == 1
    assert fib(7) == 13
    with pytest.raises(AssertionError):
        fib(-10)
