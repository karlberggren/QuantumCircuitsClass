#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:34:28 2020

@author: pmbpanther
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, nquad
from scipy import interpolate
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

    ARB: should the wavevector object also contain information about the x domain on which it is defined?
    Also, should we implement a resampling method? It might be useful to be able to quickly change the number of sample 
    points in the wavevector. 
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

    def resample_wv(self, **kwargs):
        """
        TODO implement method for dimensions greater than 1. 
        TODO implement check for limits 
        TODO zero padding if out of limits
        This method uses interpolation to resample the wavevector from x_min_new to x_max_new with N_new smaples. 
        Require: x_min_new >= x_min, x_max_new <= x_max
    
        Both up sampling and down sampling are supported. Interpolation methods: linear and spline. 
        For example:
        >>> x = np.linspace(-1,1,10)
        >>> wv_orig = Wavevector(np.exp(-x**2), [(-1,1,10)])
        >>> wv_resampled1 = wv_orig.resample_wv(range=[(-1,1,15)], method="spline")
        >>> wv_resampled1
        Wavevector([0.36787944+0.j, 0.47937337+0.j, 0.60050503+0.j,
                    0.72144476+0.j, 0.83216746+0.j, 0.92158095+0.j,
                    0.97978967+0.j, 0.99990946+0.j, 0.97978967+0.j,
                    0.92158095+0.j, 0.83216746+0.j, 0.72144476+0.j,
                    0.60050503+0.j, 0.47937337+0.j, 0.36787944+0.j])

        >>> wv_resampled2 = wv_orig.resample_wv(range=[(-1,1,6)], method="linear")
        >>> wv_resampled2
        Wavevector([0.36787944+0.j, 0.69677656+0.j, 0.95057386+0.j,
                    0.95057386+0.j, 0.69677656+0.j, 0.36787944+0.j])
        """
        method = kwargs["method"]
        domain = kwargs["range"]

        # get the current mesh grid
        array_list = []
        for x_min, x_max, N in self.ranges:
            array_list.append(np.linspace(x_min, x_max, N))
        X = np.meshgrid(*array_list)
        # get the new mesh grid
        array_list = []
        for x_min, x_max, N in domain:
            array_list.append(np.linspace(x_min, x_max, N))
        X_new = np.meshgrid(*array_list)
        
        if len(domain) == 1:
          if method == "linear":
            f = interpolate.interp1d(*X,self)
            new_obj = self.__class__(f(*X_new), domain)
            return new_obj

          elif method == "spline":
            real_tck = interpolate.splrep(*X, np.real(self), s=0)
            real_f = interpolate.splev(*X_new, real_tck, der=0)
            img_tck = interpolate.splrep(*X, np.imag(self), s=0)
            img_f = interpolate.splev(*X_new, img_tck, der=0)
            new_obj = self.__class__(real_f +1j*img_f, domain)
            return new_obj

        elif len(domain) == 2:
          if method == "linear":
            z = interpolate.griddata(np.vstack([X[0].ravel(), X[1].ravel()]).T, self, X_new, method='linear')
            new_obj = self.__class__(z, domain)
            return new_obj
          elif method == "spline":
            raise NotImplementedError


    def functionfy(self, *args):
        """
        TODO implement method.
        This method interpolates the wavevector samples and returns a wavefunction function/object defined on the domain of the wavevector
        """

        pass

    def visualize1D(self, **kwargs):   #just name it visualize?
      """
      plot_wavevector:: plot a one-dimensional wavevector.

      This is intended to be a utility function, so that, for example, one can quickly plot
      something as part of another routine.  It can also be used to simply plot a static
      wavevector.

      Not implemented: use it to update artists for an animation or another graphic setting
      that is already constructed.  

      self: wavevector to be plotted, can be a func or a vector
      range: tuple with min-max to be plotted
      N: number of plotpoints
      method: cartesian, polar, pdf, or 3d
      """ 

      x_label = kwargs["x_label"]
      method = kwargs["method"]

      # be flexible about accepting self either as a function or a vector
      if not "x_range" in kwargs:
        for x_min, x_max, N in self.ranges:
            xs = np.linspace(x_min, x_max, N)
        try:
          ψs = self
        except:
          raise NotImplementedError
      else:
        raise NotImplementedError
        x_min, x_max = kwargs["x_range"]
        N = kwargs["N"]
        xs = np.linspace(x_min, x_max, N)
        ψs = self.resample_wv(range=((x_min,x_max,N),), method="linear")

      if method == "cartesian":    
        plt.plot(xs, np.abs(ψs), label="|ψ|")
        plt.plot(xs, np.real(ψs), label="real part")
        plt.plot(xs, np.imag(ψs), label="imaginary part")
        plt.legend(loc='upper right')
        plt.xlabel(x_label)
        plt.ylabel("Amplitude")
        plt.title("Wavefunction")
        return plt.gcf()

      if method == "polar":
        # or equivalently, one can look at magnitude and phase
        # this just plots phase, is this a bug?
        plt.plot(xs, np.angle(ψs), label="phase")
        plt.xlabel(x_label)
        plt.ylabel("∠ψ")
        plt.title("Phase")
        return plt.gcf()
    
      if method == "pdf":
        plt.plot(xs, np.abs(ψs)**2, color="black")
        plt.xlabel(x_label)
        plt.ylabel("|ψ|²")
        plt.title("Prob. dens. func.")
        return plt.gcf()
    
      if method == "3d":
        #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
        fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
        ax = fig.gca(projection='3d')

        # Prepare arrays x, y, z
        y = np.imag(ψs)
        z = np.real(ψs)
        ax.plot(xs, y, z, label='parametric curve', color="red")

        # Plot a curves using the x and y axes.
        ax.plot(xs, y, zs=-1, zdir='z', label='imag part')

        # Plot a curves using the x and z axes.
        ax.plot(xs, z, zs=1, zdir='y', label='real part')

        # Plot pdf using the x and z axes
        z = np.abs(ψs)
        ax.plot(xs, z, zs=1, zdir='y', label='|Ψ|', color='black')

        x = [xs[0], xs[-1]]
        y = [0,0]
        z = [0,0]
        ax.plot(x, y, z, label='axis')

        ax.legend()
        plt.rcParams['legend.fontsize'] = 10
        return plt.gcf()


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
    from wavefunction import Wavefunction
    import doctest
    doctest.testmod()

    x = np.asarray([1. + 0.j,2,3])
    wv1 = Wavevector(x)
    assert str(wv1) == '[1.+0.j 2.+0.j 3.+0.j]', "Didn't define wavevector class correctly"

    assert str(wv1 + wv1) == '[2.+0.j 4.+0.j 6.+0.j]', "Can't add two wavevectors"
    assert str(3 + wv1) == '[4.+0.j 5.+0.j 6.+0.j]', "Can't add a constant to a wavevector"

    wf2 = Wavefunction.init_gaussian((0, 1))*1j
    wv2 = Wavevector.from_wavefunction(wf2, (-4, 4, 40))
    wf3 = wv2.resample_wv(range=((-3,3, 45),), method="linear")
    plot_params = {"method": "pdf", "x_label": "Q"}
    plt.close()
    plot_result = wv2.visualize1D(**plot_params)
    plot_result.savefig("wavevector_plot_test_file_new.png")
    plt.close()
    plot2 = wf3.visualize1D(**plot_params)
    plot2.savefig("wavevector_plot_test_file_resampled.png")
    from matplotlib.testing.compare import compare_images
    try:
        assert not compare_images("wavevector_plot_test_file.png", "wavevector_plot_test_file_new.png", .01),"Error plotting wv"
    except AssertionError:
        print("AssertionError: Error plotting wv")
    finally:
        import os
        os.remove("wavevector_plot_test_file_new.png")
    print("end")
