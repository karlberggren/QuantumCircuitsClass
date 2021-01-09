#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 17:34:28 2020

@author: pmbpanther
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

π = np.pi
oo = np.inf

class Wavefunction(object):
    """
    Class for 6.S079 Quantum Circuits, designed to work with and perform simple manipulations
    of wavefunctions, e.g. add, subtract, divide, multiply.
    """
            
    def __init__(self, wfunc):
        """
        Initializes an instance of the Wavefunction class, given a function object.
        wfunc must return a complex NumPy value.
        """
        self.ψ = wfunc
        
    def __call__(self, *args):
        return self.ψ(*args)
    
    @classmethod
    def init_gaussian(cls, *args) -> "Wavefunction object based on Gaussian":
        """
        Factory method that initializes a properly normalized Gaussian wavefunction.
        *args is a list of tuples.  Each tuple contains (Xo, σ) for one of the
        dimensions along which the gaussian is to be defined
        """
        def result(*x):
            return_val = 1
            for i, arg in enumerate(args):
                Xo, σ = arg
                print(f"x[i] type is {type(x[i])}")
                print(f"Xo type is {type(Xo)}")
                return_val *= np.exp(-(x[i] - Xo)**2/(4*σ**2))/(2*π*σ**2)**0.25+0j
            return return_val
        return cls(result)
                   
    @classmethod
    def init_plane_wave(cls, *args) -> "Wavefunction object based on plane wave":
        """
        Factory method that initializes a plane wave
        """
        def result(*x):
            return_val = 1
            for i, arg in enumerate(args):
                Xo, λ = arg
                return_val *= np.exp(1j*(x[i]-Xo)*2*π/λ)
            return return_val
        return cls(result)


    @classmethod
    def init_interp(cls, ψ_vec):
        raise NotImplementedError
    
    def __add__(self, wf2):
        return self.__class__(lambda *x: self(*x) + wf2(*x))


    def __sub__(self, wf2):
        return self.__class__(lambda *x: self(*x) - wf2(*x))


    def __mul__(self, arg2):
        """
        Multiply wavefunction by another wavefunction or a complex value
        """
        if isinstance(arg2, self.__class__):
            return self.__class__(lambda *x: self(*x) * arg2(*x))
        else:
            return self.__class__(lambda *x: arg2 * self(*x))
        
    def __rmul__(self, arg2):
        """
        Multiply wavefunction by another wavefunction or a complex value
        """    
        return self.__class__(lambda *x: arg2 * self(*x))

    
    def __truediv__(self, arg2):
        """
        Divide wavefunction by another wavefunction or a complex value
        """
        if not isinstance(arg2, self.__class__):
            return self.__class__(lambda *x: self(*x) / arg2)
        else:
            return self.__class__(lambda *x: self(*x) / arg2(*x))


    def vectorize(self, *args):
        """
        Assigns the internal variable ψ_vec to be equal to the Wavefunction's ψ, broadcasted over an array,
        startingat x_min and ending at x_max, with N total points.

        Each dimension is spec'd in an (xmin, xmax, N) tuple
        """
        # make arrays
        array_list = []
        for x_min, x_max, N in args:
            array_list.append(np.linspace(x_min, x_max, N))
        X = np.meshgrid(*array_list)
        self.ψ_mat = self(*X)
        self.ranges = args
        self.ψ_vec = self.ψ_mat.copy().flatten()
        return self.ψ_vec

    def plot_wf(self, **kwargs):
      """
      plot_wavefunction:: plot a one-dimensional wavefunction.

      This is intended to be a utility function, so that, for example, one can quickly plot
      something as part of another routine.  It can also be used to simply plot a static
      wavefunction.

      Not implemented: use it to update artists for an animation or another graphic setting
      that is already constructed.  

      self: wavefunction to be plotted, can be a func or a vector
      range: tuple with min-max to be plotted
      N: number of plotpoints
      method: cartesian, polar, pdf, or 3d
      """
      x_min, x_max = kwargs["x_range"]
      xs = np.linspace(x_min, x_max, kwargs["N"])
      x_label = kwargs["x_label"]
      method = kwargs["method"]

      # be flexible about accepting self either as a function or a vector
      try:
        ψs = self(xs)
      except:
        raise NotImplementedError

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

        x = [x_min, x_max]
        y = [0,0]
        z = [0,0]
        ax.plot(x, y, z, label='axis')

        ax.legend()
        plt.rcParams['legend.fontsize'] = 10
        return plt.gcf()

class Ket(Wavefunction):
    def __init__(self, wf):
        Wavefunction.__init__(self, wf)

    def __mul__(self, arg2):
        if arg2.__class__ == Bra:
            raise NotImplementedError("This code cannot multiply Kets times Bras.  You probably did this in error")
        else:
            return Wavefunction.__mul__(self, arg2)
        
    def __rmatmul__(self, op1):
        """
        multiply operator by a ket, returning a ket, e.g.
        O @ ket1 = ket2
        """
        return op1(self)

class Bra(Wavefunction):
    def __init__(self, wf):
        Wavefunction.__init__(self,wf)

    def __mul__(self, arg2):
        if arg2.__class__ == Ket:
            raise NotImplementedError("""
            '*' does not multiply Bra's times Ket's.  If you want to do this, use '@'.
            A Bra is effectively a row vector, and a Ket is effectively a column vector,
            so their product is effectively a dot product (i.e. a matrix operation).
            """)
        else:
            return Wavefunction.__mul__(self, arg2)

    def __matmul__(self, ket):
        # first with 1d functions
        ## FIXME: turn this into a try: to check if is 1d function
        return quad(lambda x:np.conj(self(x))*ket(x), -oo, oo)
    
if __name__ == '__main__':
    from numpy.testing import assert_approx_equal
    # test init gaussian
    for cls_type in [Wavefunction, Ket, Bra]:
        wf2 = cls_type.init_plane_wave((0, 1))
        wf1 = cls_type.init_gaussian((0, 1))
        assert wf1(0) == 1/(2*π)**0.25+0j, "Error creating gaussian"
        # test init plane wave
        assert wf2(0) == 1+0j, "Error creating plane wave"
        # test multiply by int
        assert (wf1*3)(0) == 3/(2*π)**0.25+0j, "Error multiplying wavefunc by an int"
        assert (3*wf1)(0) == 3/(2*π)**0.25+0j, "Error multiplying wavefunc by an int"
        # test mult by float
        assert (wf1*3.5)(0) == 3.5/(2*π)**0.25+0j, "Error multiplying by a float"
        # test mult by wf
        assert (wf1*wf2)(0) == 1/(2*π)**0.25+0j, "Error multiplying two wfs"
        # test div by int
        assert (wf1/2)(0) == .5/(2*π)**0.25+0j, "Error dividing by int" 
        # test div by float
        assert (wf1/0.5)(0) == 2/(2*π)**0.25+0j, "Error dividing by float"
        # test div by wf
        assert (wf1/wf2)(0) == 1/(2*π)**0.25+0j, "Error dividing wf by wf"
        # test add two wfs
        assert (wf1+wf2)(0) == 1+1/(2*π)**0.25+0j, "Error adding wf and wf"
        # test sub two wfs
        assert (wf1-wf2)(0) == 1/(2*π)**0.25-1+0j, "Error subtracting wf from wf"
        # test vectorization of wfs
        wf1.vectorize((-10, 10, 21))
        assert wf1.ψ_mat[10] == 1/(2*π)**0.25+0j, "Error vectorizing wf"

        #tests init 2d gaussian
        wf3 = cls_type.init_gaussian((0,1),(0,2))    
        assert wf3(0,0) == 1/(2*π)**0.25/(2*π*4)**.25, "Error creating gaussian"
        # test multiply by int   
        assert_approx_equal(np.real((wf3*3)(0,0)), 3/(2*π)**0.25/(2*π*4)**.25,
                            err_msg = "Error multiplying wavefunc by an int")
        # test mult by float
        assert_approx_equal(np.real((wf3*3.5)(0,0)), 3.5/(2*π)**0.25/(2*π*4)**.25,
                            err_msg = "Error multiplying by a float")
        #tests init 2d plane wave
        wf4 = cls_type.init_plane_wave((0,1),(0,2))
        assert wf4(0,0) == 1+0j, "Error creating plane wave"
        # test 2d vectorization
        wf3.vectorize((-10, 10, 21),(-10, 10, 21))
        assert wf3.ψ_mat[10][10] == 1/(2*π)**0.25/(2*π*4)**.25, "Error vectorizing 2d wf"
        #make test cases for vectorization in vectorize method.n
        assert wf3.ψ_vec[220] == 1/(2*π)**0.25/(2*π*4)**.25, "Error vectorizing 2d wf"
        wf3.vectorize((-10, 10, 41),(-10, 10, 21))
        assert wf3.ψ_mat[10][20] == 1/(2*π)**0.25/(2*π*4)**.25, "Error vectorizing 2d wf"
        """
        # test div by int
        assert (wf3/2)(0,0) == .5/(2*π)**0.25/(2*π*4)**.25, "Error dividing by int"    
        # test div by float
        assert (wf3/0.5)(0,0) == 2/(2*π)**0.25/(2*π*4)**.25, "Error dividing by float"
        """

        wf1 = cls_type.init_gaussian((0, 1))*1j
        plot_params = {"x_range": (-4, 4), "N": 40,
                       "method": "pdf", "x_label": "Q"}
        plt.close()
        plot_result = wf1.plot_wf(**plot_params)
        plot_result.savefig("wavefunction_plot_test_file_new.png")
        from matplotlib.testing.compare import compare_images
        try:
            assert not compare_images("wavefunction_plot_test_file.png", "wavefunction_plot_test_file_new.png", .001),"Error plotting wf"
        except AssertionError:
            print("AssertionError: Error plotting wf")
        finally:
            import os
            os.remove("wavefunction_plot_test_file_new.png")

    for cls1,cls2 in ((Bra, Ket), (Ket, Bra)):        
        wf2 = cls1.init_plane_wave((0, 1))
        wf1 = cls2.init_gaussian((0, 1))
        try :
            wf1 * wf2
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{cls2} * {cls1} worked, shouldn't have")

        
    """
    test 3 D figure plot

    # plot_params["method"] = "cartesian"
    # plot_params["method"] = "polar"
    plot_params["method"] = "3d"
    plt.close()
    wf1.plot_wf(**plot_params)
    plt.show()
    """
    print("Ended Wavefunction run")

    wf2 = Ket.init_gaussian((0, 1))
    wf1 = Bra.init_gaussian((0, 1))
    wf1 @ wf2
