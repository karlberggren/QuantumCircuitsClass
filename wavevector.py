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

    >>> x = np.asarray([1. + 0.j,2,3])
    >>> wv1 = Wavevector(x, [(-1, 1, 3)])
    >>> print(wv1)
    [1.+0.j 2.+0.j 3.+0.j]
    >>> print(wv1.ranges)
    [(-1, 1, 3)]
    """
    
    def __new__(cls, input_array, ranges = None):
        obj = np.asarray(input_array).view(cls).astype(complex)
        obj.ranges = ranges
        return obj


    def __array_finalize__(self, obj):
        """
        'obj' is the numpy object that is viewcast or
        template cast from, in order to generate the 
        current instance
        e.g.
        a = np.arange(10)
        cast_a = a.view(Wavevector)
        in this case obj is the "a" object, and self
        is the cast_a instance, which will be of type
        Wavevector.

        When created from an explicit constructor 'obj'
        is None.
        """
        # this handles the case when created from initialization call
        if obj is None:
            return
        # and now if created from template or view cast
        self.ranges = getattr(obj, 'ranges', None)
        
    @classmethod
    def from_wavefunction(cls, wf, *args):
        """
        Factory method that takes a Wavefunction and a sequence of tuples (one for each 
        dimension of the Wavefunction)
        and creates a discrete N-dimensional array in the Wavevector class.

        Each dimension is spec'd in an (xmin, xmax, N) tuple.

        For example, vectorizing a gaussian might look like

        >>> wf = Wavefunction.init_gaussian((0,1))
        >>> wv = Wavevector.from_wavefunction(wf, (-1, 1, 3))
        >>> print(wv)
        [0.4919052 +0.j 0.63161878+0.j 0.4919052 +0.j]
        >>> print(wv.ranges)
        ((-1, 1, 3),)
        """
        # make arrays
        array_list = []
        for x_min, x_max, N in args:
            array_list.append(np.linspace(x_min, x_max, N))
        X = np.meshgrid(*array_list)
        new_wavevector = cls(wf(*X))
        new_wavevector.ranges = args
        return new_wavevector
    def meshify(self):
        """
        >>> wf = Wavefunction.init_gaussian((0,1), (0,2))
        >>> wv = Wavevector.from_wavefunction(wf, (-1,1,3), (-1,1,3))
        >>> wv.meshify()
        [array([[-1.,  0.,  1.],
               [-1.,  0.,  1.],
               [-1.,  0.,  1.]]), array([[-1., -1., -1.],
               [ 0.,  0.,  0.],
               [ 1.,  1.,  1.]])]
        """
        return np.meshgrid(*((np.linspace(x_min, x_max, N) for x_min, x_max, N in self.ranges)))
    
if __name__ == '__main__':
    import doctest
    doctest.testmod()

    x = np.asarray([1. + 0.j,2,3])
    wv1 = Wavevector(x)
    assert str(wv1) == '[1.+0.j 2.+0.j 3.+0.j]', "Didn't define wavevector class correctly"

    assert str(wv1 + wv1) == '[2.+0.j 4.+0.j 6.+0.j]', "Can't add two wavevectors"
    assert str(3 + wv1) == '[4.+0.j 5.+0.j 6.+0.j]', "Can't add a constant to a wavevector"
    print("end")
