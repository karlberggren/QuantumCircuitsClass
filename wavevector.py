#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:34:28 2020

@author: pmbpanther
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, nquad
from numpy.testing import assert_almost_equal
from inspect import signature

π = np.pi
oo = np.inf

class Wavevector(np.ndarray):
    """
    Class for 6.S079 Quantum Circuits, designed to work with and perform simple manipulations
    of wavevectors (the discretized equivalent of wavefunctions), e.g. add, subtract, divide, multiply.
    Please note that a continuous wavefunction is still a vector in an infinite-dimensional space
    (a Hilbert space), and a properly normalized wavefunction would be a unit vector in such a space.
    """
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls).astype(complex)
        return obj

    def __array_finalize__(self, obj):
        """
        'obj' is the numpy object that is viewcast or
        template cast from, in order to generate the 
        current instance
        e.g.
        >>> a = np.arange(10)
        >>> cast_a = a.view(Wavevector)
        in this case obj is the "a" object, and self
        is the cast_a instance, which will be of type
        Wavevector.

        When created from an explicit constructor 'obj'
        is None.
        """
        
        pass

if __name__ == '__main__':
    x = np.asarray([1. + 0.j,2,3])
    wv1 = Wavevector(x)
    assert str(wv1) == '[1.+0.j 2.+0.j 3.+0.j]', "Didn't define wavevector class correctly"

    assert str(wv1 + wv1) == '[2.+0.j 4.+0.j 6.+0.j]', "Can't add two wavevectors"
    assert str(3 + wv1) == '[4.+0.j 5.+0.j 6.+0.j]', "Can't add a constant to a wavevector"
    print("end")
