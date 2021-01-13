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
    def __new__(cls, input_array, ranges = None):
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

    @classmethod
    def from_wavefunction(cls, wf, *args):
        """
        Factory method that takes a Wavefunction and a sequence of tuples (one for each dimension of the Wavefunction)
        and creates a discrete N-dimensional array in the Wavevector class.

        Each dimension is spec'd in an (xmin, xmax, N) tuple.

        For example, vectorizing a gaussian might look like

        >>> Wavevector.from_wavefunction(Wavefunction.init_gaussian((0,1)), (-10, 10, 100))

        """
        # make arrays
        array_list = []
        for x_min, x_max, N in args:
            array_list.append(np.linspace(x_min, x_max, N))
        X = np.meshgrid(*array_list)
        new_wavevector = cls(wf(*X))
        #new_wavevector.ranges = args
        #new_wavevector.vec = new_wavevector.copy().flatten()
        return new_wavevector


    

if __name__ == '__main__':
    x = np.asarray([1. + 0.j,2,3])
    wv1 = Wavevector(x)
    assert str(wv1) == '[1.+0.j 2.+0.j 3.+0.j]', "Didn't define wavevector class correctly"

    assert str(wv1 + wv1) == '[2.+0.j 4.+0.j 6.+0.j]', "Can't add two wavevectors"
    assert str(3 + wv1) == '[4.+0.j 5.+0.j 6.+0.j]', "Can't add a constant to a wavevector"
    print("end")
