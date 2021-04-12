"""enables working with quantum wavefunctions

Work with, evolve, and plot wavevectors

  Typical usage:

  FIXME put example here

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, nquad
from scipy import interpolate
from numpy.testing import assert_almost_equal
from inspect import signature
from scipy.integrate import solve_ivp
from q_operator import Op_matx
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button, CheckButtons
import random

pause = False

π = np.pi
oo = np.inf
ħ = 1.05e-34 
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
    #def from_wf(cls, wf: Wavefunction, *args):
    def from_wf(cls, wf, *args):
        """make wavevector from wavefunction

        Factory method that takes a Wavefunction and a sequence of tuples (one for each 
        dimension of the Wavefunction)
        and creates a discrete N-dimensional array in the Wavevector class.

        Args:
            wf: wavefunction to be converted into wavevector
            *args: iterator of tuples, where each dimension is spec'd in an (xmin, xmax, N) tuple.

        For example, vectorizing a gaussian might look like

        >>> wf = Wavefunction.init_gaussian((0,1))
        >>> wv = Wavevector.from_wf(wf, (-1, 1, 3))
        >>> print(wv)
        [0.4919052 +0.j 0.63161878+0.j 0.4919052 +0.j]
        >>> print(wv.ranges)
        ((-1, 1, 3),)

        Returns:
            New n-dimensional wavevector with appropriately defined ranges etc.
        """
        # make arrays
        array_list = []
        for x_min, x_max, N in args:
            array_list.append(np.linspace(x_min, x_max, N))
        X = np.meshgrid(*array_list)
        new_wavevector = cls(wf(*X))
        new_wavevector.ranges = args
        return new_wavevector

    def simple_measure_1d(self, M: int, seed: int = 0):
        """collapse wavefunction into a subspace

        Perform a simulated measurement on the wavevector that projects it into
        a simple subspace and then renormalizes the output to return the post-measurement
        wavevector.

        The subspaces will just be those spanned by adjacent groups of the 
        full function's unit vectors.  e.g. if function is defined 
        
        Args:
            M: number of subspaces onto which measurement is projected

        Returns:
            Wavevector consisting of normalized post-measurement

        >>> wf = Wavefunction.init_gaussian((0,1))
        >>> wv = Wavevector.from_wf(wf, (-1, 1, 4))
        >>> print(wv.simple_measure_1d(2))
        [0. +0.j 0. +0.j 0.5+0.j 0.5+0.j]

        >>> wf2 = Wavefunction.init_gaussian((1,0.5))
        >>> wv2 = Wavevector.from_wf(wf2, (-1, 3, 16))
        >>> print(wv2.simple_measure_1d(4))
        [0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.25+0.j 0.25+0.j 0.25+0.j 0.25+0.j
         0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j 0.  +0.j]

        >>> wf3 = Wavefunction.init_gaussian((1,0.1))
        >>> wv3 = Wavevector.from_wf(wf2, (-0.5, 2, 12))
        >>> print(wv3.simple_measure_1d(6))
        [0. +0.j 0. +0.j 0. +0.j 0. +0.j 0. +0.j 0. +0.j 0. +0.j 0. +0.j 0.5+0.j
         0.5+0.j 0. +0.j 0. +0.j]
        """
        # set the seed to get predictable results
        np.random.seed(seed) 
        # initiate a table of probabilities to store the probability of the flux being found in each region
        probability_table = []
        # get del x
        for xmin, xmax, N in self.ranges:
            delx = (xmax - xmin)/(N-1) 
        # for every region:
        for i in range(M):
            inds = [j for j in range(round(i*len(self)/M), round((i+1)*len(self)/M))]
            exclude_inds = list(set(range(len(self))) - set(inds))
            # create a  projection matrix x:
            x = np.identity(len(self), dtype=np.float32)*delx
            x[exclude_inds, exclude_inds] = 0
            # find probability of flux being in that region by taking <phi^* | x | phi> and store in the probability table 
            prob = np.real(np.transpose(np.conjugate(self)) @ x @ self)
            probability_table.append(prob)
        # Use multinomial RV to get the resul of throwing a weighted cube. Multinomial returns an array of size p.size where the entry in each index is the number of times
        # the cube landed on that face
        probability_table = np.array(probability_table)/np.sum(probability_table)   # normalize probabilities in case wavevctor isn't normalized
        cube_throw = np.random.multinomial(1, probability_table)
        region_number = int((np.where(cube_throw ==1)[0][0]))  # for some odd reason numpy returns the array index s a float which needs to be converted to an int for indexing
        inds = [j for j in range(round(region_number*len(self)/M), round((region_number+1)*len(self)/M))]
        exclude_inds = list(set(range(len(self))) - set(inds))

        # collaps the wavefunction 
        self[inds] = 1
        self[exclude_inds] = 0
        # normalize it
        self /= sum(self)
        return self

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
        This method interpolates the wavevector samples and returns a wavefunction function/object 
        defined on the domain of the wavevector
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
        fig, (ax1, ax2) = plt.subplots(2)
        fig.suptitle('Polar plot')
        ax1.plot(xs, np.abs(ψs), label="magnitude")
        ax1.set(ylabel="|ψ|")
        ax2.plot(xs, np.angle(ψs), label="phase")
        ax2.set(xlabel=x_label, ylabel="∠ψ")
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
        >>> wv = Wavevector.from_wf(wf, (-1,1,3), (-1,1,3))
        >>> wv.meshify()
        [array([[-1.,  0.,  1.],
               [-1.,  0.,  1.],
               [-1.,  0.,  1.]]), array([[-1., -1., -1.],
               [ 0.,  0.,  0.],
               [ 1.,  1.,  1.]])]
        """
        return np.meshgrid(*((np.linspace(x_min, x_max, N) for x_min, x_max, N in self.ranges)))


    # def evolve(self, Vfunc,
    #            masses: tuple,
    #            times: tuple,
    #            frames: int = 30,
    #            t_dep: bool = True) -> np.array:
    def evolve(self, Vfunc, masses, times, frames = 30, t_dep = True):
        """evolves wavevector in a (possibly time_varying) potential.

        Evolves the wavevector, changing its value continuously in time, and 
        storing in a history array the value at certain snapshots.

        Args:
            Vfunc: A potential energy function
            masses: list of tuples containing m_eff for each dimension. 
                    Thus for a 1D function, it should be of the form (m1,)

            times: tuple in form "start time", "end time"

            frames: number of frames to record the evolution at

            t_dep: boolean specifies whether Vfunc is time dependent

        >>> dim_info = ((-2, 2, 5),)
        >>> masses = (ħ,)
        >>> wv_o = Wavevector.from_wf(Wavefunction.init_gaussian((0,1)), *dim_info)
        >>> r = wv_o.evolve(lambda x: x-x, masses, (0, 1e-32), frames = 3, t_dep = False)
        >>> print(r.y)
        [[2.32359563e-01+0.j 4.82278714e-04+0.j 1.62011520e-06+0.j]
         [4.91905199e-01+0.j 8.41415942e-04+0.j 3.18509494e-08+0.j]
         [6.31618778e-01+0.j 9.64557428e-04+0.j 3.24023040e-06+0.j]
         [4.91905199e-01+0.j 8.41415942e-04+0.j 3.18509494e-08+0.j]
         [2.32359563e-01+0.j 4.82278714e-04+0.j 1.62011520e-06+0.j]]
        """ 
        if t_dep:
            def dψdt(t, ψ):  # key function for evolution
                return (KE.dot(ψ) + params["V_matx"]*ψ)/(1j*ħ)
            raise NotImplementedError
        else:
            # make our Hamiltonian
            KE_args = [val + (m,) for val, m in zip(self.ranges, masses)]
            KE = Op_matx.make_KE(*KE_args)
            # don't need effective mass for potential arguments, so strip away mass part
            # from KE_args
            potential = Op_matx.from_function(Vfunc, *self.ranges)
            Hamiltonian = KE + potential

            def dψdt(t, ψ):  # key function for evolution
                return Hamiltonian.dot(ψ)/(1j*ħ)

            # peform simulation
            frame_times = np.linspace(*times, frames)
            r = solve_ivp(dψdt, times, self, method='RK23', 
                          t_eval = frame_times)

            if not (r.status == 0):  # solver did not reach the end of tspan
                print(r.message)

            r.ranges = self.ranges
            return r

    def _evolve_frame_1d(self):
        """
        """
        return


    def realtime_evolve(self, Vfunc, masses, timescale, n = 20, t_dep = True, method = "pdf"):
        """
        evolves a wavefunction in "real" time.  Permits pausing and resuming.

        timescale::characteristic time scale like period or decay time, will take
                   about 2π seconds to run
        n::framerate at which you wish to run the simulation, in frames/sec
        """
        # first set up the axes, leaving some room for widgets
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.25)
        
        # add a pause button
        axcolor = 'white'
        ax_pause = plt.axes([0.775, 0.825, 0.1, 0.04])  # define loc and dim of pause button
        pause_button = Button(ax_pause, 'Pause', color=axcolor, hovercolor='lightblue')
        pause = False  # initiate state unpaused
        pause_dict = {False: "Pause", True: "Resume"}


        def pause_event(event):
            nonlocal pause
            pause ^= True  # toggle pause status
            pause_button.label.set_text(pause_dict[pause])

        pause_button.on_clicked(pause_event)
        
        # set duration of each frame in simulation time
        Δt = timescale/(2*π*n)
        t_o = 0

        #
        new_wv_lst = [Wavevector(self, self.ranges)]  # horrible kludge to pass reference
        new_wv = Wavevector(self, self.ranges)

        def anim_func(i):
            nonlocal t_o, pause, new_wv

            if pause:  # skip out of pause button pressed
                return
            t_f = t_o + Δt  # increment end time
            # run simulation

            r = new_wv.evolve(Vfunc, masses, (t_o, t_f), frames = 2, t_dep = t_dep)
            t_o = t_f  # prepare for next step in simulation

            # plot
            ax.cla()  # first clear old axes
            if method == "pdf":
                ax.plot(np.linspace(*new_wv.ranges[0]), np.abs(r.y.T[-1])**2, color='tab:blue')
            elif method == "polar":
                raise NotImplementedError
            ax.set_ylabel('|ψ²|')
            ax.set_xlabel('Q or Φ')
            new_wv = Wavevector(r.y.T[-1], new_wv.ranges)

        ani = FuncAnimation(fig, anim_func, interval = 1000//n)
        return ani, pause_button
        

class Evolution(object):
    """ class that gathers an array of wavevectors with identical
    range data 

    FIXME, I really wonder if this shouldn't be a function instead of
    a class
    """

    def __init__(self, wvs, time_range, frames):
        self.ranges = wvs[0].ranges
        self.time_range = time_range
        self.frames = frames
        self.wvs = wvs

    def visualize_1D(self):
        raise NotImplementedError

    

if __name__ == '__main__':
    from wavefunction import Wavefunction
    import doctest
    doctest.testmod()

    x = np.asarray([1. + 0.j, 2, 3])
    wv1 = Wavevector(x)
    assert str(wv1) == '[1.+0.j 2.+0.j 3.+0.j]', "Didn't define wavevector class correctly"

    assert str(wv1 + wv1) == '[2.+0.j 4.+0.j 6.+0.j]', "Can't add two wavevectors"
    assert str(3 + wv1) == '[4.+0.j 5.+0.j 6.+0.j]', "Can't add a constant to a wavevector"

    """
    wf2 = Wavefunction.init_gaussian((0, 1))*1j
    wv2 = Wavevector.from_wf(wf2, (-4, 4, 40))
    wf3 = wv2.resample_wv(range=((-3,3, 45),), method="linear")
    plot_params = {"method": "polar", "x_label": "Q"}
    plt.close()
    plot_result = wv2.visualize1D(**plot_params)
    plt.show()
    plot_result.savefig("wavevector_plot_test_file_new.png")
    plt.close()
    plot2 = wf3.visualize1D(**plot_params)
    plot2.savefig("wavevector_plot_test_file_resampled.png")
    from matplotlib.testing.compare import compare_images
    """
#    try:
#        assert not compare_images("wavevector_plot_test_file_oldest.png", "wavevector_plot_test_file_new.png", 10),"Error plotting wv"
#    except AssertionError:
#        print("AssertionError: Error plotting wv")
#    finally:
#        import os
#        os.remove("wavevector_plot_test_file_new.png")

             
    dim_info = ((-20, 20, 200),)
    masses = (ħ,)
    wv_o = Wavevector.from_wf(Wavefunction.init_gaussian((0,1)), *dim_info)
    ani, button = wv_o.realtime_evolve(lambda x: x-x, masses, 1e-33, n=4, t_dep = False)
    plt.show()
    print("end wavevector")
    
