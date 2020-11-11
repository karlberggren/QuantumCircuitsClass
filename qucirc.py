import numpy as np

import sympy as sym

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rc
from mpl_toolkits.mplot3d import Axes3D

import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy import interpolate
import scipy.sparse as sparse
from scipy.integrate import complex_ode
from scipy.integrate import ode
from scipy.misc import derivative

from IPython.display import HTML

ħ = 1  # h = 6.63e-34 J s or 6.58e-16 eV s
       # ħ = h / 2π = 1.05 e -34
π = np.pi

def make_gaussian(Qo: "Center value", σ: "Standard Deviation"):
    """
    Qo : the center of the distribution
    σ : the standard deviation of the final PDF distrubtion
    """
    return lambda Q: (2*π*σ**2)**(0.25) * np.exp(-(Q - Qo)**2/(4*σ**2)) + 0j

def make_plane_wave(φ: "Phase", k: "Radial spatial frequency"):
  return lambda Q: np.exp(1j * (k * Q - φ))

def funcsum(func1, func2):
    def helper(x):
      return 1/np.sqrt(2)*(func1(x) + func2(x))
    return helper

def funcmult(const, func):
  """ helper closure that multiplies a function by a constant """
  def helper(x):
    return const*func(x)
  return helper

def mult_funcs(func1, func2):
  """ helper that multiplies two functions """
  return lambda Q: func1(Q) * func2(Q)

from ipywidgets import interactive
import matplotlib.pyplot as plt
import numpy as np

def f(Qo, σ):
    plt.figure(2)
    x = np.linspace(-10, 10, num=1000)
    plt.plot(x, np.abs(make_gaussian(Qo, σ)(x))**2)
    plt.ylim(-10, 10)
    plt.show()

interactive_plot = interactive(f, Qo=(-5, 5.0,.5), σ=(0.5, 5, .5))
output = interactive_plot.children[-1]
output.layout.height = '350px'
interactive_plot

def plot_wavefunction(func, range = (-4,4), N = 40, method="cartesian"):
  """
  plot_wavefunction
  func: wavefunction to be plotted
  range: tuple with min-max to be plotted
  N: number of plotpoints
  method: cartesian, polar, pdf, or 3d
  """
  min, max = range
  Q = np.linspace(min, max, N)
  if method == "cartesian":
    plt.plot(Q, np.abs(func(Q)), label="|ψ|")
    plt.plot(Q, np.real(func(Q)), label="real part")
    plt.plot(Q, np.imag(func(Q)), label="imaginary part")
    plt.legend(loc='upper right')
    plt.xlabel("Q")
    plt.ylabel("Amplitude")
    plt.title("Wavefunction")
    plt.show()

  if method == "polar":
    # or equivalently, one can look at magnitude and phase
    plt.plot(Q, np.angle(func(Q)), label="phase")
    plt.xlabel("Q")
    plt.ylabel("phase")
    plt.title("Phase")
    plt.show()

  if method == "pdf":
    plt.plot(Q, np.abs(func(Q))**2)
    plt.xlabel("Q")
    plt.ylabel("|ψ|²")
    plt.title("Prob. dens. func.")
    plt.show()

  if method == "3d":
    #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
    ax = fig.gca(projection='3d')

    # Prepare arrays x, y, z
    y = np.imag(func(Q))
    z = np.real(func(Q))
    ax.plot(Q, y, z, label='parametric curve',color="red")
    #print(x,y,z)

    # Plot a curves using the x and y axes.
    ax.plot(Q, y, zs=-1, zdir='z', label='imag part')

    # Plot a curves using the x and z axes.
    ax.plot(Q, z, zs=1, zdir='y', label='real part')

    # Plot pdf using the x and z axes
    z = np.abs(func(Q))
    ax.plot(Q, z, zs=1, zdir='y', label='|Ψ|', color='black')
    
    x = [min, max]
    y = [0,0]
    z = [0,0]
    ax.plot(x, y, z, label='axis')

    ax.legend()
    plt.rcParams['legend.fontsize'] = 10
    plt.show()

  if method == "3d_arrows":
    #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
    ax = fig.gca(projection='3d')

    # Prepare arrays x, y, z
    y = np.imag(func(Q))
    z = np.real(func(Q))
    ax.plot(Q, y, z, label='parametric curve',color="red")
    #print(x,y,z)

    # Plot a curves using the x and y axes.
    ax.plot(Q, y, zs=-1, zdir='z', label='imag part')

    # Plot a curves using the x and z axes.
    ax.plot(Q, z, zs=1, zdir='y', label='real part',color='black')


    x = [min, max]
    y = [0,0]
    z = [0,0]
    ax.plot(x, y, z, label='axis')

    num_arrows = N
    x = np.linspace(min, max, num_arrows)
    y, z = np.zeros(num_arrows), np.zeros(num_arrows)
    u = np.zeros(num_arrows)
    v = np.imag(func(x))
    w = np.real(func(x))
    ax.quiver(x, y, z, y, v, w, arrow_length_ratio=0.15)

    ax.legend()
    plt.rcParams['legend.fontsize'] = 10
    plt.show()

def plot_time_dep_ψ(func, 
                    params, 
                    method="animate_2d",
                    num_pts = 100):
  """
  plot_time_dep_ψ: plot wavefunction in time and position

  arguments:
  func -- function to be plotted, takes 2 parameters, (Q,t)
  params -- parameters
  method -- pdf: probability density function in 2 d plot
            animate_2d: 2d animation of pdf
            animate_3d: 3d animation of wavefunction.
  num_pts -- number of points to be plotted along x axis
  """
  Qmin, Qmax = params["Q_min"], -params["Q_min"]
  Q_range = np.linspace(Qmin, Qmax, params["N"])
  frames = params["frames"]

  if method == "pdf":
    for time in make_time_series(params):
      state = func(Q_range, time)
      plt.plot(Q_range, np.abs(state)**2, label=f"{time:.2f}")
    plt.legend(loc = 'best')
    plt.xlabel("Charge (Q)")
    plt.ylabel("PDF")
    plt.show()
    return
  
  if method == "animate_2d":
    fig = plt.figure()
    new_data = np.array([func(Q_range, t) for t in make_time_series(params)])
    ymax = np.max(np.abs(new_data)**2)
    ax = fig.add_subplot(111, xlim = (Qmin, Qmax), ylim = (0,ymax))
    #ax.plot(i_L, 1/2*params["L"]*i_L**2)
    particles, = ax.plot([], [])
    ax.set_ylabel("PDF")
    ax.set_xlabel("charge (Q)")
    def animate(i):
        """perform animation step"""
        data = new_data[i]
        data = np.abs(data)**2
        particles.set_data([Q_range],[data])
        return particles,

    ani = animation.FuncAnimation(fig, animate, frames=new_data.shape[0], interval=100,
                                  blit=True)
    rc('animation', html='jshtml')
    return ani

  if method == "animate_3d":
    #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
    ax = fig.gca(projection='3d')

    # set plot limits somewhat automatically
    ax.set_xlim((Qmin, Qmax))
    x = np.linspace(Qmin, Qmax, num_pts)
    times = make_time_series(params)
    max = 1.5 * np.max(np.abs([[func(Q, t) for Q in x] for t in times]))
    ax.set_ylim((-max, max))
    ax.set_zlim((-max, max))

    # artist for main parametric curve
    line1 = ax.plot([], [], [], lw=2, label='parametric curve',color="red")[0]
    
    # artists for real and imaginary projections
    line2 = ax.plot([],[], zs=-max, zdir='z', label='imaginary part')[0]
    line3 = ax.plot([], [], zs=max, zdir='y', label='real part')[0]

    # artist for pdf
    line4 = ax.plot([], [], zs=max, zdir = 'y', label='|ψ|')[0]
    
    lines = [line1,line2,line3,line4]  # array of artists for udpate

    # animation function. This is called sequentially  
    def update_lines(i, lines):
        line1, line2, line3,line4 = lines
        
        Q = np.linspace(Qmin, Qmax, num_pts)
        ωo, T, duration, dt = extract_times(params)
        t_start = 0
        dt = (duration)/(frames-1)
        t = t_start + i*dt
        
        # update main parametric line
        y = np.imag(func(Q, t))
        z = np.real(func(Q, t))
        line1.set_data(Q, y)
        line1.set_3d_properties(z)

        # projection onto imag axis
        z = np.full(num_pts,-max)
        line2.set_data(Q,y)
        line2.set_3d_properties(z)

        # projection onto real axis
        z = np.real(func(Q,t))
        y = np.full(num_pts,max)
        line3.set_data(Q,z)
        line3.set_3d_properties(y,zdir='y')
        
        # plot pdf
        z = np.abs(func(Q,t))
        line4.set_data(Q,z)
        #line4.set_3d_properties()
        

        return (line1, line2, line3,line4)

    x = [Qmin, Qmax]
    y = [0,0]
    z = [0,0]
    ax.plot(x, y, z, label='axis')

    ax.legend()
    anim = animation.FuncAnimation(fig, update_lines,
                                  fargs=[lines], frames=frames, interval=100,
                                  blit=True)
    rc('animation', html='jshtml')  # makes it work in colaboratory
    return anim

def plot_wavefunction_discrete(vec, params, N=100, method="cartesian"):
  min, max = params["Q_min"], -params["Q_min"]
  Q = make_Qrange(params)
  if method == "cartesian":
    plt.plot(Q, np.abs(vec), label="|ψ|")
    plt.plot(Q, np.real(vec), label="real part")
    plt.plot(Q, np.imag(vec), label="imaginary part")
    plt.legend(loc='upper right')
    plt.xlabel("Q")
    plt.ylabel("Amplitude")
    plt.title("Wavefunction")
    plt.show()

  if method == "polar":
    # or equivalently, one can look at magnitude and phase
    plt.plot(Q, np.angle(vec), label="phase")
    plt.xlabel("Q")
    plt.ylabel("phase")
    plt.title("Phase")
    plt.show()

  if method == "pdf":
    plt.plot(Q, np.abs(vec)**2)
    plt.xlabel("Q")
    plt.ylabel("|ψ|²")
    plt.title("Prob. dens. func.")
    plt.show()

  if method == "3d":
    scale = np.amax(np.abs(vec))
    ymax = scale*1.5
    zmax = scale*1.5
    
    #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
    ax = fig.gca(projection='3d')

    # set limits
    ax.set_ylim(-ymax, ymax)
    ax.set_zlim(-zmax, zmax)

    # Prepare arrays x, y, z
    y = np.imag(vec)
    z = np.real(vec)
    ax.plot(Q, y, z, label='parametric curve',color="red")
    #print(x,y,z)

    # Plot a curves using the x and y axes.
    ax.plot(Q, y, zs=-zmax, zdir='z', label='imag part')

    # Plot a curves using the x and z axes.
    ax.plot(Q, z, zs=zmax, zdir='y', label='real part')

    # Plot pdf using the x and z axes
    z = np.abs(vec)
    ax.plot(Q, z, zs=zmax, zdir='y', label='|Ψ|', color='black')
    
    x = [min, max]
    y = [0,0]
    z = [0,0]
    ax.set_xlabel("Charge")
    ax.set_ylabel("imag(ψ)")
    ax.set_zlabel("real(ψ)(orange), |ψ|(black)")
    ax.plot(x, y, z, label='axis')

    ax.legend()
    plt.rcParams['legend.fontsize'] = 10
    plt.show()

def extract_times(params):
  ωo = 1/np.sqrt(params["L"] * params["C"])
  T = 2*π/ωo
  duration = T * params["periods"]
  Δt = duration/params["frames"]
  return ωo, T, duration, Δt

def make_Qrange(params):
  return np.linspace(params["Q_min"], -params["Q_min"], params["N"])

def make_time_series(params):
  ωo, T, duration, Δt= extract_times(params)
  return np.linspace(0, duration, params["frames"])

def expectation_value(O: "Operator",
                      ψ: "Wavefunction",
                      params):
  if isinstance(ψ, np.ndarray) and sparse.issparse(O):
    norm = np.real(np.transpose(np.conj(ψ)) @ ψ)
    return np.real(np.transpose(np.conj(ψ)) @ (O @ ψ))/norm
  else:
    ψ_star = lambda *args: np.conj(ψ(*args))
    i_func = lambda *args: np.real(ψ_star(*args) * O(ψ)(*args))
    return integrate.quad(i_func, params["Q_min"], -params["Q_min"])[0]

def exp_vs_t(O: "Operator",
             data_or_func: "array or function", 
             params,
             show_plot = True):
  """ 
  ψ: function or numpy array
  O: operator or numpy array.  If operator, should take a function as an input
  
  If inputs are both numpy arrays, O should be a matrix that performs
  the desired operation and a simple expectation value is calculated.  ψ should
  be an array of time-step arrays, i.e. [ψ(0), ψ(dt), ...]
  
  If inputs are both functions, an integral is performed
  """
  t_series = make_time_series(params)
  if isinstance(data_or_func, np.ndarray) and sparse.issparse(O):
    data = data_or_func
    expect_Q = [expectation_value(O, datum, params) for datum in data]
  else:
    func = data_or_func
    expect_Q=[]
    for t in t_series:
      newfunc = lambda Q: func(Q,t)
      expect_Q.append(expectation_value(O, newfunc, params))
  if show_plot:
    plt.plot(t_series, expect_Q)
    plt.xlabel("time [s]")
    plt.ylabel("⟨ψ|O|ψ⟩")
    plt.show()
  return expect_Q

def Φ_hat(ψ, Q):
  return lambda Q: -1j*ħ*derivative(ψ, Q, dx=1e-6)  # the dx is a kludge, magic number, FIXME

def Q_hat(ψ, Q):
  return lambda Q: Q*ψ(Q)

def make_Φ_hat(params, method='symmetric'):
  ΔQ = -2*params["Q_min"]/params["N"]
  if method == 'symmetric':
    Φhat = - 1j * ħ * sparse.diags([[-1], 
                                   np.ones(params["N"]-1),
                                   -1*np.ones(params["N"]-1),
                                   [1]],
                                   [params["N"]-1, 1,-1, -params["N"]+1]) / (2 * ΔQ)
  else:
    Φhat = - 1j * ħ * sparse.diags([[-1], 
                                   np.ones(params["N"]),
                                   -1*np.ones(params["N"]-1)],
                                   [params["N"]-1, 0,-1]) / ΔQ

  return Φhat

def make_Φ_hat_squared(params):
  N = params["N"]
  ΔQ = -2*params["Q_min"]/N
  Φhat_sq = - ħ**2 * sparse.diags([np.full(N,-2),
                                   np.ones(N-1),
                                   np.ones(N-1),
                                   [1],[1]],[0,1,-1,N-1,-(N-1)])/ΔQ**2
  return Φhat_sq

def make_Q_hat(params):
  Q_range = make_Qrange(params)
  return sparse.diags(Q_range)

def make_identity(params):
  return sparse.diags(np.ones(params["N"]))

def wf_to_func(rout_or_func, params, method="1d"):
  Q_range = make_Qrange(params)
  
  if method == "1d":
    func = rout_or_func
    return interpolate.interp1d(Q_range, func(Q_range))

  if method == "2d":
    rout = rout_or_func
    ωo, T, duration, Δt= extract_times(params)
    data = np.transpose(rout.y)
    re_func = interpolate.interp2d(Q_range, rout.t, np.real(data), kind='linear')
    im_func = interpolate.interp2d(Q_range, rout.t, np.imag(data), kind='linear')
    return lambda x,t: re_func(x,t) + 1j*im_func(x,t)

def ivp_evolve(params):
  """
  evolves a probability distribution in a quantum L-C circuit with a
  time-dependent potential.
  """ 
  params["Q_period"] = -2 * params["Q_min"]

  ωo = np.sqrt(1/params["L"]/params["C"])
  
  T = 2*π/ωo
  params["start_time"] = 0
  params["end_time"] = params["start_time"] + params["periods"] * T
  params["dt"]= (params["end_time"] - params["start_time"])/params["frames"] 
  params["ΔQ"] = params["Q_period"]/params["N"]
  t0, t1, dt = params["start_time"], params["end_time"], params["dt"]
  Q_min, Q_period, N = params["Q_min"], params["Q_period"], params["N"]
  Q_range = np.linspace(Q_min, Q_min + Q_period, N)
  ψo = params["wavefunction"](Q_range)

  def make_matrix(params):
    """ make finite-element matrix for simulation """
    # diagonals
    N = params["N"]
    off = np.ones(N - 1)  # sparse matrices are defined by diagonal arrays
    mid = np.full(N, -2)  # middle
    corn = np.ones(1)  # corners
    L, ΔQ = params["L"], params["ΔQ"]
    return sparse.diags([corn,off,mid,off,corn],[-(N-1),-1,0,1,(N-1)])*1j*ħ/(2*L*ΔQ**2)

  params["matx"] = make_matrix(params)
  C = params["C"]

  V = params["potential"]
  
  Q_range = np.linspace(Q_min, Q_period + Q_min, N)
  params["Q_range"] = Q_range

  def dψdt(t, ψ, params): 
    return params["matx"].dot(ψ) - 1j/ħ * V(params["Q_range"], t, params) * ψ

  times = np.linspace(t0, t1, params["frames"])
  r = solve_ivp(dψdt, (t0, t1), ψo, method='RK23', t_eval = times, args = (params,))
  #print(r.t)
  if not (r.status == 0):  # solver did not reach the end of tspan
    print(r.message)
    
  return r, wf_to_func(r, params, method="2d")

def plot_time_dep_V(params, 
                    method="animate",
                    num_pts = 100):
  """
  plot_time_dep_V: plot potential in time and charge

  arguments:
  func -- function to be plotted, takes 2 parameters, (Q,t)
  params -- parameters
  method -- pdf: probability density function in 2 d plot
            animate_2d: 2d animation of pdf
            animate_3d: 3d animation of wavefunction.
  num_pts -- number of points to be plotted along x axis
  """
  func = params["potential"]
  Qmin, Qmax = params["Q_min"], -params["Q_min"]
  Q_range = np.linspace(Qmin, Qmax, params["N"])
  frames = params["frames"]

  if method == "2d":
    for time in make_time_series(params):
      state = func(Q_range, time, params)
      plt.plot(Q_range, state, label=f"{time:.2f}")
    plt.legend(loc = 'best')
    plt.xlabel("Charge (Q)")
    plt.ylabel("V(Q)")
    plt.show()
    return
  
  if method == "animate":
    fig = plt.figure()
    new_data = np.array([func(Q_range, t, params) for 
                         t in make_time_series(params)])
    ymax = np.max(new_data)
    ax = fig.add_subplot(111, xlim = (Qmin, Qmax), ylim = (0,ymax))
    particles, = ax.plot([], [])
    ax.set_ylabel("V(Q)")
    ax.set_xlabel("charge (Q)")
    def animate(i):
        """perform animation step"""
        data = new_data[i]
        particles.set_data([Q_range],[data])
        return particles,

    ani = animation.FuncAnimation(fig, animate, frames=new_data.shape[0], interval=100,
                                  blit=True)
    rc('animation', html='jshtml')
    return ani

def hilbert_dot(vec1, vec2):
  # calculate the hilbert dot product of two vectors
  return np.conj(np.transpose(vec1))@vec2

def make_vec(func, params):
  # utility function that helps turn a function into a vector
  Qrange = make_Qrange(params)
  return func(Qrange)

def find_coefficients(func, basis, params):
  """
  decompose vector into a set of basis vectors, and return a list of the
  relative weights/coefficients of each basis vector
  """
  vec = make_vec(func, params)
  coeffs = []
  for elt in basis:
    coeffs.append(hilbert_dot(elt,vec))
  return coeffs

def solve_se(evecs, evals, coeffs, t, params):
  Qrange = make_Qrange(params)
  comps = []
  for coeff,evec,eval in zip(coeffs,evecs,evals):
    comps.append(coeff*evec*np.exp(-1j*eval*t/ħ))
  return np.sum(comps,0)
