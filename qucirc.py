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
from scipy.optimize import fsolve

from IPython.display import HTML


from scipy.special import factorial
from scipy.special import hermite

import matplotlib.patches as patches
from matplotlib import cm


ħ = 1  # h = 6.63e-34 J s or 6.58e-16 eV s
       # ħ = h / 2π = 1.05 e -34
π = np.pi
𝕛 = 1j
Φₒ = 1  # Φₒ = 2.07e-15 V s
Φₒbar = Φₒ/2/π

"""
Created on Fri Jul 12 14:04:23 2019
@author: artmenlope
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection



def fill_between_3d(ax,x1,y1,z1,x2,y2,z2,mode=1,c='steelblue',alpha=0.6):
    
    """
    
    Function similar to the matplotlib.pyplot.fill_between function but 
    for 3D plots.
       
    input:
        
        ax -> The axis where the function will plot.
        
        x1 -> 1D array. x coordinates of the first line.
        y1 -> 1D array. y coordinates of the first line.
        z1 -> 1D array. z coordinates of the first line.
        
        x2 -> 1D array. x coordinates of the second line.
        y2 -> 1D array. y coordinates of the second line.
        z2 -> 1D array. z coordinates of the second line.
    
    modes:
        mode = 1 -> Fill between the lines using the shortest distance between 
                    both. Makes a lot of single trapezoids in the diagonals 
                    between lines and then adds them into a single collection.
                    
        mode = 2 -> Uses the lines as the edges of one only 3d polygon.
           
    Other parameters (for matplotlib): 
        
        c -> the color of the polygon collection.
        alpha -> transparency of the polygon collection.
        
    """

    if mode == 1:
        
        for i in range(len(x1)-1):
            
            verts = [(x1[i],y1[i],z1[i]), (x1[i+1],y1[i+1],z1[i+1])] + \
                    [(x2[i+1],y2[i+1],z2[i+1]), (x2[i],y2[i],z2[i])]
            
            ax.add_collection3d(Poly3DCollection([verts],
                                                 alpha=alpha,
                                                 linewidths=0,
                                                 color=c))

    if mode == 2:
        
        verts = [(x1[i],y1[i],z1[i]) for i in range(len(x1))] + \
                [(x2[i],y2[i],z2[i]) for i in range(len(x2))]
                
        ax.add_collection3d(Poly3DCollection([verts],alpha=alpha,color=c))


def make_gaussian(Qo: "Center value", σ: "Standard Deviation"):
    """
    Qo : the center of the distribution
    σ : the standard deviation of the final PDF distrubtion
    """
    return lambda Q: (2*π*σ**2)**(-0.25) * np.exp(-(Q - Qo)**2/(4*σ**2)) + 0j

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



def plot_time_dep_ψ(func_or_data, 
                    params, 
                    method="animate_2d",
                    num_pts = 100):
  """
  plot_time_dep_ψ: plot wavefunction in time and position

  arguments:
  func_or_data -- function to be plotted, takes 2 parameters, (Q,t), or
                  array of data as a array of wavefunction arrays at each
                  time frame
  params -- parameters
  method -- pdf: probability density function in 2 d plot
            animate_2d: 2d animation of pdf
            animate_3d: 3d animation of wavefunction.
  num_pts -- number of points to be plotted along x axis
  """
  # some setup for use in the various methods
  try:
    Qmin, Qmax = params["min"], params["max"]
  except KeyError:
    Qmin, Qmax = params["min"], -params["min"]
    
  Q_range = np.linspace(Qmin, Qmax, params["N"])
  frames = params["frames"]
  try:
    times = np.linspace(params["start_time"],
                        params["end_time"],
                        frames)
  except KeyError:
    raise NotImplementedError("I broke the old period-based way of timing FIXME")

  if method == "pdf":
    for time in make_time_series(params):
      try:
        state = func_or_data(Q_range, time)
      except :
        raise NotImplementedError('pdf with data type not implemented')
      finally :
        plt.plot(Q_range, np.abs(state)**2, label=f"{time:.2f}")
        plt.legend(loc = 'best')
        plt.xlabel("Charge (Q)")
        plt.ylabel("PDF")
        plt.show()
    return
  
  if method == "animate_2d":
    fig = plt.figure()
    
    try:
      new_data = np.array([func_or_data(Q_range, t)
                           for t in make_time_series(params)])
    except:
      new_data = func_or_data
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

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=new_data.shape[0],
                                  interval=100,
                                  blit=True)
    #rc('animation', html='jshtml')
    return ani

  """
  #
  # 2d_eigen:
  #
  # method that plots time-varying potential along with wavefunction
  #
  """
  if method == "2d_eigen":
    fig = plt.figure()
    
    try:
      new_data = np.array([func_or_data(Q_range, t)
                           for t in make_time_series(params)])
    except:
      new_data = func_or_data

    # set scales
    pdfmax = np.max(np.abs(new_data)**2)
    V = params["potential"]
    vmin = np.min([[V(t, Q, params) for t in times] for Q in Q_range])  
    vmax = np.max([[V(t, Q, params) for t in times] for Q in Q_range])

    # set up plots and artists
    ax = fig.add_subplot(111, xlim = (Qmin, Qmax), ylim = (vmin,vmax))
    ax.set_ylabel("PDF")
    ax.set_xlabel("phase (φ)")

    particles, = ax.plot([], [])
    pot_parts, = ax.plot([], [])
    artists = [particles, pot_parts]

    # show the time 0 potential in the background
    ax.plot(Q_range, V(0,Q_range,params))

    # show the eigenvalues of the time 0 potential in the background
    # FIXME, THIS NEEDS TO BE GENERALIZED FOR ALL POTENTIALS!
    """
    for eval in [ħ*ωo*(2*n+1)/2 for n in range(5)]:
      #Q_min,Q_max = find_turning_points(V,E)
      ax.plot(Q_range, eval, "--")      
    """
    
    def animate(i: "Step", artists: "List of artists to animate"):
        """perform animation step"""
        # some setup
        Q = np.linspace(Qmin, Qmax, num_pts)
        particles, pot_arts = artists
        
        # pdf.  
        try:  # did we pass a function?
          z = np.abs(func_or_data(Q,t))**2
        except:  # no, raw data
          z = np.abs(func_or_data[i])**2

        # resample array to get length right
        if params["N"] % num_pts == 0:
          z = z[::params["N"]//num_pts]
        else:
          raise Except("params[N] must be an integer multiple of num_pts")

        # rescale array so it looks nice
        z *= 0.5*vmax/pdfmax
        z += vmin
        particles.set_data([Q],[z])

        # potential
        duration = params["end_time"] - params["start_time"]
        t = params["start_time"] + duration/params["frames"]*i
        V = params["potential"]

        z = V(t, Q, params)
        pot_parts.set_data(Q, z)
        
        # update potential
    
        return particles,pot_parts

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=new_data.shape[0],
                                  fargs=[artists],
                                  interval=100,
                                  blit=True)
    #rc('animation', html='jshtml')
    return ani

  """
  #
  # 2d_eigen_full:
  #
  # method that plots time-varying potential along with wavefunction
  # and the various eigenstates, weighted by their occupancy
  #
  """
  if method == "2d_eigen_full":
    fig = plt.figure()
    
    try:
      new_data = np.array([func_or_data(Q_range, t)
                           for t in make_time_series(params)])
    except:
      new_data = func_or_data

    # set scales
    pdfmax = np.max(np.abs(new_data)**2)
    V = params["potential"]
    vmax = np.max([[V(t, Q, params) for t in times] for Q in Q_range])

    # set up plots and artists
    ax = fig.add_subplot(111, xlim = (Qmin, Qmax), ylim = (0,vmax))
    ax.set_ylabel("V(φ)")
    ax.set_xlabel("phase (φ)")   

    particles, = ax.plot([], [], color = "silver")
    pot_parts, = ax.plot([], [], color = "tab:orange")
    artists = [particles, pot_parts]

    # show the time 0 potential in the background
    ax.plot(Q_range, V(0,Q_range,params))

    # show the eigenvalues of the time 0 potential in the background
    # FIXME, THIS NEEDS TO BE GENERALIZED FOR ALL POTENTIALS!
    def find_turning_points(xs, V: "potential function", E: "energy"):
      """
      find_turning_points:
      xs:: range of x values along which to search
      V:: potential function
      E: energy values to find turning points
      """
      # find left hand turning point xmin
      xmin = xs[0]
      xmax = xs[-1]
      for x in xs:
          if V(x) < E:
              xmin = x
              break
      # left hand t.p. found, now start from right
      for x in xs[::-1]:
          if V(x) < E:
              xmax = x
              #print(f"found max at x={xmax}")
              break   
      return xmin, xmax
    
    ωo = 1/np.sqrt(params["C"]*params["Lo"])

    def sho_evec(n, params):  # FIXME replace with esystem
        m = params["C"]
        ω = 1/np.sqrt(params["C"]*params["Lo"])
        norm = 1/np.sqrt(2**n * factorial(n))*(m*ω/π/ħ)**0.25
        #E_val = ħ*ω*(2*n+1)/2
        ψ = lambda Q: norm * np.exp(-m*ω*Φₒ**2*Q**2/4/π**2/2/ħ)*hermite(n)(Φₒ*Q/2/π) #* np.exp(1j*E_val*t/ħ)
        return ψ

    shadings=[]
    n_max = int((vmax - ħ*ωo/2)/ħ/ωo)
    print(n_max)
    
    for n in range(n_max):
      eval = ħ*ωo*(2*n+1)/2
      Q_min, Q_max = find_turning_points(Q_range,
                                         lambda Q: V(0, Q, params),
                                         eval)
      eval_range = np.linspace(Q_min, Q_max)
      ax.plot(eval_range, np.full(50, eval), "--", lw = 0.5, 
              color='tab:blue')
      evec = sho_evec(n, params)(Q_range)
      offset = np.full(np.shape(Q_range),eval)
      new_artist, = ax.plot([], [], lw = 0.5, color='tab:blue')
      v = np.array([[0,0]])
      artists.append(patches.Polygon(v, closed=True, fc='tab:blue', ec='tab:blue', alpha=0.3))
      ax.add_patch(artists[-1])
      #artists.append(new_artist)
      #ax.plot(Q_range, np.abs(evec)**2*vmax/pdfmax*0.025+offset,color='black')
    
    
    def animate(i, artists):
        """perform animation step"""
        # some setup
        Q = np.linspace(Qmin, Qmax, num_pts)
        particles, pot_arts, evecs = artists[0], artists[1], artists[2:]
        
        # pdf.  
        try:  # did we pass a function?
          zo = np.abs(func_or_data(Q,t))**2
          zraw = func_or_data(Q,t)
        except:  # no, raw data
          zo = np.abs(func_or_data[i])**2
          zraw = func_or_data[i]

        # resample array to get length right
        if params["N"] % num_pts == 0:
          z = zo[::params["N"]//num_pts]
        else:
          raise Except("params[N] must be an integer multiple of num_pts")

        # rescale array so it looks nice
        z *= 0.5*vmax/pdfmax
        particles.set_data([Q],[z])

        # potential
        duration = params["end_time"] - params["start_time"]
        t = params["start_time"] + duration/params["frames"]*i
        V = params["potential"]

        z = V(t, Q, params)
        pot_parts.set_data(Q, z)
        
        # evecs
        base_clock = 0
        # project current wf onto evec bases
        for n, evec in enumerate(evecs):
          double_well_params = params.copy()
          double_well_params["Lo"] = Φₒbar / double_well_params["Ic"]
          evec_array = sho_evec(n, double_well_params)(Q_range)
          evec_array /= np.sqrt(np.transpose(np.conj(evec_array))@evec_array)
          coeff = np.transpose(np.conj(evec_array))@zraw
          eval = ħ*ωo*(2*n+1)/2
          offset = np.full(np.shape(Q_range),eval)
          scale = 0.5*vmax/pdfmax
          if coeff > 0.1:
            yvals = np.abs(coeff*evec_array)**2*scale+offset
            xvals = Q
            vs = np.column_stack((xvals,yvals))
            evec.set_xy(vs)
            #color stuff not implemented
            #cmap = cm.get_cmap('Spectral')
            #if n == 0:
            #  base_clock = coeff
            #rgba = cmap(np.angle(coeff - base_clock)/π+0.5)
            #evec.set_fc(rgba)
            #evec.set_ec(rgba)
            
            #evec.set_data(Q, np.abs(coeff*evec_array)**2*scale+offset)

    
        return [particles, pot_parts] + evecs

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  frames=new_data.shape[0],
                                  fargs=[artists],
                                  interval=100,
                                  blit=True)
    #rc('animation', html='jshtml')
    return ani

  if method == "animate_3d":
    #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
    ax = fig.gca(projection='3d')

    # set plot limits somewhat automatically
    ax.set_xlim((Qmin, Qmax))
    x = np.linspace(Qmin, Qmax, num_pts)
    times = make_time_series(params)
    try:
      max = 1.5 * np.max(np.abs([[func_or_data(Q, t)
                                  for Q in x]
                                 for t in times])**2)
    except:
      max = 1.5 * np.max(np.abs(func_or_data)**2)
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
        try:
          z = np.real(func_or_data(Q,t))
        except:
          z = np.real(func_or_data[i])
        y = np.full(num_pts,max)
        line3.set_data(Q,z)
        line3.set_3d_properties(y,zdir='y')
        
        # plot pdf
        try:
          z = np.abs(func_or_data(Q,t))
        except:
          z = np.abs(func_or_data[i])**2
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
    #rc('animation', html='jshtml')  # makes it work in colaboratory
    return anim

  if method == "3d_eigen":
    """
    makes a plot designed to show eigensystem evolving and changes in
    occupancy of eigenstates as system evolves
    """
    Q = make_Qrange(params)

    #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    fig = plt.figure(figsize=plt.figaspect(0.5)*1.5)
    ax = fig.gca(projection='3d')

    # set plot limits somewhat automatically
    ax.set_xlim((Qmin, Qmax))
    x = np.linspace(Qmin, Qmax, num_pts)
    times = np.linspace(params["start_time"],
                        params["end_time"],
                        params["frames"])
    try:
      max = 1.5 * np.max(np.abs([[func_or_data(Q, t)
                                  for Q in x]
                                  for t in times])**2)
    except:
      max = 1.5 * np.max(np.abs(func_or_data)**2)
    
    ax.set_ylim(params["potential"](0, 0, params),
                params["potential"](0, Qmax, params))
    ax.set_zlim((0, max))

    # artist for potential curve
    line1 = ax.plot([], [], [],
                    lw=2,
                    zdir='y',
                    label='potential',
                    color='red')[0]

    # artists for real and imaginary projections
    line2 = ax.plot([], [], zs=-max, zdir='z', label='imaginary part')[0]
    line3 = ax.plot([], [], zs=max, zdir='y', label='real part')[0]

    # artist for pdf
    line4 = ax.plot([], [], zs=max, zdir = 'y', label='|ψ|²')[0]
    
    #lines = [line1,line2,line3,line4]  # array of artists for udpate
    lines = [line1, line4]


    # Draw lines for eigenenergies
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    """
    make eigenfunctions of potential
    n = 0
    while ħ * ωo * (2*n + 1)/ 2 < params["Q_min"]**2/2/params["C"]:
      E_val = ħ * ωo * (2*n + 1)/ 2
      Q_val = np.sqrt(2 * params["C"] * E_val)

      # Plot a curves using the x and z axes. 
      #ax.plot(Q, z, zs=1, zdir='y', label='real part')
      def func(Q, t, params):  # FIXME replace with esystem
        m = params["L"]
        ω = 1/np.sqrt(params["C"]*params["L"])
        norm = 1/np.sqrt(2**n * factorial(n))*(m*ω/π/ħ)**0.25
        ψ = norm * np.exp(-m*ω*Q**2/2/ħ)*hermite(n)(Q) * np.exp(1j*E_val*t/ħ)
        return ψ

      # plot eigenvalues
      x = [-Q_val, Q_val]
      ax.plot([-Q_val, Q_val],
              [E_val, E_val],
              zs=0,
              zdir='z',
              color='green')

      try:
        y = np.full(Q.shape, E_val) + np.imag(func_or_data(Q,t,params))
      except:
        y = np.full(Q.shape, E_val) + np.imag(func_or_data[0]))  #FIXME

      try:
        z = np.real(func(Q, t, params))
      except:
        z = np.real(func_or_data[0])  #FIXME
      # draw outer line of eigenstate
      ax.plot(Q, y, z, zdir='z', color='blue')
      # bottom line for fill
      y_bottom = np.full(Q.shape, E_val)
      fill_between_3d(ax,
                      Q,
                      y_bottom,
                      np.full(Q.shape,0),
                      Q,
                      y,
                      z,
                      mode=1,
                      c="C0")
      
      n += 1
      
    ax.set_xlabel("Charge")
    ax.set_ylabel("Energy")
    if method == 'ψ':
      ax.set_zlabel("ψ")
    elif method == 'pdf':
      ax.set_zlabel("|ψ|²")
      
    ax.set_zlim(0,3)
    ax.legend()
    ax.view_init(elev=45, azim=45)
    plt.rcParams['legend.fontsize'] = 10
    plt.show()
      
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

    #ax.legend()  FIXME
    #plt.rcParams['legend.fontsize'] = 10
        # animation function. This is called sequentially
    """
    def update_lines(i, lines, params):
        line1, line4 = lines
        
        Q = np.linspace(Qmin, Qmax, num_pts)
        duration = params["end_time"] - params["start_time"]
        t = params["start_time"] + duration/params["frames"]*i
        V = params["potential"]
        
        # update potential
        z = V(t, Q, params)
        #print(y)
        y = np.zeros(Q.shape)
        # Plot curves using the x and y axes.
        #ax.plot(Q, y, zs=0, zdir='z', label='potential')
        line1.set_data(Q, y)
        line1.set_3d_properties(z)

        """y = np.imag(func(Q, t))
        z = np.real(func(Q, t))
        line1.set_data(Q, y)
        line1.set_3d_properties(z)

        # projection onto imag axis
        z = np.full(num_pts,-max)
        line2.set_data(Q,y)
        line2.set_3d_properties(z)

        # projection onto real axis
        try:
          z = np.real(func_or_data(Q,t))
        except:
          z = np.real(func_or_data[i])
        y = np.full(num_pts,max)
        line3.set_data(Q,z)
        line3.set_3d_properties(y,zdir='y')
        """
        #plot pdf
        try:
          z = np.abs(func_or_data(Q,t))**2
        except:
          z = np.abs(func_or_data[i])**2
        # resample array
        if params["N"] % num_pts == 0:
          z = z[::params["N"]//num_pts]
        else:
          raise Except("params[N] must be an integer multiple of num_pts")
        line4.set_data(Q, z)
        line4.set_3d_properties(z)

        return (line1, line4)
        #line2, line3,line4)
    
    anim = animation.FuncAnimation(fig,
                                   update_lines,
                                   fargs=[lines, params],
                                   frames=frames, interval=100,
                                   blit=True)
    #rc('animation', html='jshtml')  # makes it work in colaboratory
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
  try:
    ωo = 1/np.sqrt(params["L"] * params["C"])
  except KeyError:
    ωo = 1/np.sqrt(params["Lo"] * params["C"])
      
  T = 2*π/ωo
  try:
    duration = T * params["periods"]
  except KeyError:
    duration = params["end_time"] - params["start_time"]
  Δt = duration/params["frames"]
  return ωo, T, duration, Δt

def make_Qrange(params):
  try:
    result = np.linspace(params["Q_min"], -params["Q_min"], params["N"])
  except KeyError:
    try:
      result = np.linspace(params["min"], params["max"], params["N"])
    except KeyError:
      result = np.linspace(params["min"],-params["min"], params["N"])

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
    #rc('animation', html='jshtml')
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

"""
New operators for phase-basis matrices.  Taken from Week 9 lecture 2020.
"""

def make_φ_range(params):
  """ make_φ_range: create appropriate x-axis vector for phase
  """
  minφ = params["min"]
  try:
    maxφ = params["max"]
  except KeyError:
    maxφ = -params["min"]
  return np.linspace(minφ, maxφ, params["N"])  

def make_φ_hat(params):
  φ_range = make_φ_range(params)
  return sparse.diags(φ_range)

def make_cos_φ_hat(params):
  φ_range = make_φ_range(params)
  return sparse.diags(np.cos(φ_range))

def make_Q_hat_squared(params):
  """ make_Q_hat_squared: construct Q² operator
  """
  N = params["N"]
  minφ = params["min"]
  try:
    maxφ = params["max"]
  except KeyError:
    maxφ = -params["min"]
  Δφ = (maxφ-minφ)/N
  #coeff = 1j*ħ/Δφ
  coeff = 𝕛*2*π*ħ/Φₒ/Δφ
  Qhat_sq = coeff**2 * sparse.diags([np.full(N,-2),
                                    np.ones(N-1),
                                    np.ones(N-1)],[0,1,-1]) #no periodic b.c.
  return Qhat_sq


def make_Q_in_φ_basis(params):
  N = params["N"]
  minφ = params["min"]
  try:
    maxφ = params["max"]
  except KeyError:
    maxφ = -params["min"]

  Δφ = (maxφ-minφ)/N
  coeff = 𝕛*2*π*ħ/Φₒ/Δφ
  Qhat = coeff * sparse.diags([np.ones(N-1),
                               np.ones(N-1)],[1,-1])/2 #no periodic b.c.
  return Qhat
    

def find_esystem(params):
  """ find_esystem: find eigensystem
  returns array of values and vector, aligned and ordered from low to high """
  Qhat_sq = make_Q_hat_squared(params)
  φhat = make_φ_hat(params)
  cos_φhat = make_cos_φ_hat(params)
  # V and KE should be passed to it in params, IMHO
  I, I_C = params["I"],params["I_C"]
  V = - Φₒ * I_C * cos_φhat / 2 / π - I * Φₒ / 2 / π * φhat
  KE = Qhat_sq/(2*params["C"])
  
  # Hamiltonian
  ℋ = KE + V
  vals, vecs = eigs(np.real(ℋ), k=6, which='SM')
  vecs = np.transpose(vecs)
  return vals, vecs

def ivp_evolve_time_dep(params):
  """
  evolves a probability distribution in a quantum circuit with a
  time_varying potential.
  """ 

  def dψdt(t, ψ, params):  # key function for evolution
    φ_range = make_φ_range(params)
    try:
      params["Qo"]
      KE = make_tdep_KE_matx(t, params)
    except:
      KE = params["KE_matx"]
    V_t = params["potential"](t, φ_range, params)
    return (KE.dot(ψ) + V_t*ψ)/(1j*ħ)

 # if "max" parameter is provided, use it, else just uses "min"
  xmin = params["min"]
  try:
    xmax = params["max"]
  except KeyError:
    xmax = -params["min"]
  params["xrange"] = xmax - xmin
  
  try:  # if "L" is provided, use it, else use "end_time"
    ωo = np.sqrt(1/params["L"]/params["C"])
    T = 2*π/ωo
    params["start_time"] = 0
    params["end_time"] = params["start_time"] + params["periods"] * T
  except KeyError:
    pass

  # peform simulation
  dt= (params["end_time"] - params["start_time"])/params["frames"] 
  t0, t1 = params["start_time"], params["end_time"]
  xmin, xrange, N = params["min"], params["xrange"], params["N"]
  xs = np.linspace(xmin, xmax, N)  # list of our x points
  ψo = params["wavefunction"](xs)  # starting wavevector
  times = np.linspace(t0, t1, params["frames"])
  r = solve_ivp(dψdt, (t0, t1), ψo, method='RK23', 
                t_eval = times, args = (params,))

  if not (r.status == 0):  # solver did not reach the end of tspan
    print(r.message)
    
  return r

def make_V_matx(t, params):
  """
  Make potential matrix using potential function
  """
  V = params["potential"]
  φ_range = make_φ_range(params)
  V_matx = sparse.diags([V(φ_range, params)],[0])
  return V_matx

def make_KE_matx(params):
  """
  Make kinetic energy matrix
  """
  Qhat_sq = make_Q_hat_squared(params)
  KE_matx = Qhat_sq/(2*params["C"])
  return KE_matx

def make_tdep_KE_matx(t, params):
  """
  Make kinetic energy matrix
  """
  ωd = params["drive_freq"]
  Qo = params["Qo"]*np.sin(ωd * t)
  Qhat_sq = make_Q_hat_squared(params)
  Qhat = make_Q_in_φ_basis(params)
  KE_matx = (Qhat_sq - 2 * Qo * Qhat)/(2*params["C"])
  return KE_matx


def ivp_evolve_time_dep_test1():
  """
  do a simple test of ivp_evolve.  static no pot'l.  The wavepacket slowly
  diffuses outwards, then wraps around at the periodic b.c., so you see a
  standing wave form in pdf.
  """
  params = {"min": -20, "I": 0, "I_C": 1, "N":2000, "C":1,
            "start_time":0, "end_time":1, "frames":100,
            "wavefunction":make_gaussian(0, 1)}
  def V(t, φ, params):
    return 0
            
  params["potential"] = V          
  params["KE_matx"] = make_KE_matx(params)
  
  r = ivp_evolve_time_dep(params)
  ani = plot_time_dep_ψ(np.transpose(r.y), 
                        params, 
                        method="animate_2d",
                        num_pts = 100)
  return ani

def ivp_evolve_time_dep_test2():
  """
  do a simple test of ivp_evolve.  static LC system.  The wavefunction
  breathes back and forth slowly.
  """
  params = {"min": -10, "I": 0, "I_C": 1, "N":1000, "C":1,
            "start_time":0, "end_time":1, "frames":100,
            "wavefunction":make_gaussian(0, 1.5),
            "Lo": 1, "α":0}
  def V(t, φ, params):
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    L_t = params["Lo"]*(1 + params["α"]*np.cos(2*ωo*t))
    return φ**2/(2*L_t)
            
  params["potential"] = V
  params["KE_matx"] = make_KE_matx(params)
  
  r = ivp_evolve_time_dep(params)
  ani = plot_time_dep_ψ(np.transpose(r.y), 
                        params, 
                        method="animate_2d",
                        num_pts = 100)
  return ani

def ivp_evolve_time_dep_test3():
  """
  test that ground state does not evolve in time
  """
  params = {"min": -20, "N":800, "C":1,
            "start_time":0, "end_time":100, "frames":185,
            "wavefunction":make_gaussian(0, 2*π/np.sqrt(2)/Φₒ),
            "Lo": 1, "α":0.00}
  def V(t, φ, params):
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    L_t = params["Lo"]*(1 + params["α"]*np.cos(2*ωo*t))
    #return φ**2/(2*L_t)
    return Φₒ**2*φ**2/(2*L_t*4*π**2)
            
  params["potential"] = V
  params["KE_matx"] = make_KE_matx(params)
  
  r = ivp_evolve_time_dep(params)
  ani = plot_time_dep_ψ(np.transpose(r.y), 
                        params, 
                        method="animate_2d",
                        num_pts = 100)
  return ani

def ivp_evolve_time_dep_test4():
  """
  test parametrically driven L-C circuit, where L changes in time
  """
  params = {"min": -20, "N":800, "C":1,
            "start_time":0, "end_time":10, "frames":200,
            "wavefunction":make_gaussian(0, 2*π/np.sqrt(2)/Φₒ),
            "Lo": 1, "α":0.1}
  def V(t, φ, params):
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    L_t = params["Lo"]*(1 + params["α"]*np.cos(2*ωo*t))
    #return φ**2/(2*L_t)
    return Φₒ**2*φ**2/(2*L_t*4*π**2)
            
  params["potential"] = V
  params["KE_matx"] = make_KE_matx(params)
  
  r = ivp_evolve_time_dep(params)
  ani = plot_time_dep_ψ(np.transpose(r.y), 
                        params, 
                        method="animate_2d",
                        num_pts = 100)
  return ani

def ivp_evolve_time_dep_test5():
  """
  test parametrically driven L-C circuit, where L changes in time,
  looking at 2d plot of potential and wavefunction.  You see higher
  order eigenstates appear to be filled.
  """
  params = {"min": -20, "N":800, "C":1,
            "start_time":0, "end_time":2*π*10, "frames":400,
            "wavefunction":make_gaussian(0, 2*π/np.sqrt(2)/Φₒ),
            "Lo": 1, "α":0.1}
  def V(t, φ, params):
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    L_t = params["Lo"]*(1 + params["α"]*np.cos(2*ωo*t))
    #return φ**2/(2*L_t)
    return Φₒ**2*φ**2/(2*L_t*4*π**2)
            
  params["potential"] = V
  params["KE_matx"] = make_KE_matx(params)
  
  r = ivp_evolve_time_dep(params)
  ani = plot_time_dep_ψ(np.transpose(r.y), 
                        params, 
                        method="2d_eigen",
                        num_pts = 100)
  return ani

def plot_q_paramp_test():
  """
  test parametrically driven L-C circuit, where L changes in time
  """
  params = {"min": -20, "N":400, "C":1,
            "start_time":0, "end_time":2*π*5, "frames":200,
            "wavefunction":make_gaussian(0, 2*π/np.sqrt(2)/Φₒ),
            "Lo": 1, "α":0.1}
  def V(t, φ, params):
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    L_t = params["Lo"]*(1 + params["α"]*np.sin(2*ωo*t))
    return Φₒ**2*φ**2/(2*L_t*4*π**2)
            
  params["potential"] = V
  params["KE_matx"] = make_KE_matx(params)
  
  r = ivp_evolve_time_dep(params)
  ani = plot_time_dep_ψ(np.transpose(r.y), 
                        params, 
                        method="2d_eigen",
                        num_pts = 100)
  
  return ani


def plot_q_paramp_test_2():
    params = {"min": -20, "N":100, "C":1,
              "start_time":0, "end_time":2*π*10, "frames":200,
              "wavefunction":make_gaussian(0, 2*π/np.sqrt(2)/Φₒ),
              "Lo": 1, "α":0.1}
    def V(t, φ, params):
      ωo = 1/np.sqrt(params["C"]*params["Lo"])
      L_t = params["Lo"]*(1 + params["α"]*np.sin(2*ωo*t))
      return Φₒ**2*φ**2/(2*L_t*4*π**2)
          
    params["potential"] = V
    params["KE_matx"] = make_KE_matx(params)

    r = ivp_evolve_time_dep(params)
    ani = plot_time_dep_ψ(np.transpose(r.y), 
                          params, 
                          method="2d_eigen_full",
                          num_pts = 100)
    return ani

def sho_evec(n, params):
  m = params["C"]
  ω = 1/np.sqrt(params["C"]*params["Lo"])
  norm = 1/np.sqrt(2**n * factorial(n))*(m*ω/π/ħ)**0.25
  ψ = lambda Q: norm * np.exp(-m*ω*Φₒ**2*Q**2/4/π**2/2/ħ)*hermite(n)(Φₒ*Q/2/π) + 0j
  #* np.exp(1j*E_val*t/ħ)
  return ψ

def superposition(params):
  return lambda Q: (sho_evec(0, params)(Q) + sho_evec(1, params)(Q))/(np.sqrt(2)) + 0j

def plot_q_paramp_test_3():
    """
    drive a state parametrically with a ground state phase distribution
    but with some initial momentum
    """
    params = {"min": -20, "N":100, "C":1,
              "start_time":0, "end_time":2*π*1, "frames":10,
              "Lo": 1, "α":0,
              "amp": 0.37,
              "Qo": 0.1
              }

    params["wavefunction"] = sho_evec(1,params)
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    params["drive_freq"] = ωo
    
    def V(t, φ, params):
      ωd = params["drive_freq"]
      L_t = params["Lo"]*(1 + params["α"]*np.sin(ωd*t))
      return Φₒ**2*(φ - params["amp"]*np.sin(ωd*t))**2/(2*L_t*4*π**2)
              
    params["potential"] = V
    params["KE_matx"] = make_KE_matx(params)

    r = ivp_evolve_time_dep(params)
    ani = plot_time_dep_ψ(np.transpose(r.y), 
                          params, 
                          method="2d_eigen_full",
                          num_pts = 100)
    return ani

def adiabatic_test():
    params = {"min": -20, "N":100, "C":1,
              "start_time":0, "end_time":2*π*10, "frames":100,
              "Lo": 1, "α":0,
              "amp": 5}
    params["wavefunction"] = sho_evec(0, params)
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    params["drive_freq"] = ωo

    def V(t, φ, params):
      tstart = 30
      tend = 31
      slew_rate = params["amp"]/(tend - tstart)
      if t <= tstart:
        φ_ext = 0
      elif tstart < t <= tend:
        φ_ext = (t - tstart) * slew_rate
      elif t > tend:
        φ_ext = (tend - tstart) * slew_rate
      ωd = params["drive_freq"]
      L_t = params["Lo"]*(1 + params["α"]*np.sin(ωd*t))
      return Φₒ**2*(φ - φ_ext)**2/(2*L_t*4*π**2)
          
    params["potential"] = V
    params["KE_matx"] = make_KE_matx(params)
    r = ivp_evolve_time_dep(params)
    
    ani = plot_time_dep_ψ(np.transpose(r.y), 
                          params, 
                          method="2d_eigen_full",
                          num_pts = 100)
    ani


def double_well_evolution():
    """
    C: Josephson capacitance
    Lo: RF SQUID inductance
    I_C: critical current
    """
    params = {"min": -4, "max":2*π + 4,
              "N":200, "C":1,
              "start_time" :0, "end_time":5, "frames":300,
              "Lo" : 1,
              "Ic" : 80,
              "Ibias" : 0,
              "φext": π}
    
    #params["wavefunction"] = sho_evec(0, params)

    soln0 = fsolve(lambda φ: (φ)/params["Lo"] - params["Ic"]*np.sin(φ), -3)[0] + π
    soln1 = fsolve(lambda φ: (φ)/params["Lo"] - params["Ic"]*np.sin(φ), 3)[0] + π
    params["wavefunction"] = make_gaussian(0.4, 1)
    ωo = 1/np.sqrt(params["C"]*params["Lo"])
    params["drive_freq"] = ωo

    def V(t, φ, params):
        Ibias, Ic, Lo, φext = params["Ibias"],params["Ic"],params["Lo"],params["φext"]
        VL = (Φₒbar*(φ-φext))**2/2/Lo
        Vbias = Ibias * φ * Φₒbar
        Vjj = -Φₒbar * Ic * np.cos(φ)
        return VL + Vbias + Vjj
          
    params["potential"] = V
    params["KE_matx"] = make_KE_matx(params)

    r = ivp_evolve_time_dep(params)
    #plt.plot(np.transpose(r.y)[1])
    #plt.show()

    ani = plot_time_dep_ψ(np.transpose(r.y), 
                          params, 
                          method="2d_eigen",
                          num_pts = 100)
    ani
    
    
if __name__=='__main__':
  #ivp_evolve_time_dep_test1()
  #ivp_evolve_time_dep_test2()
  #ivp_evolve_time_dep_test3()    
  #ivp_evolve_time_dep_test4()
  #ivp_evolve_time_dep_test5()
  #plot_q_paramp_test_2()
  #plot_q_paramp_test_3()
  #adiabatic_test()
  double_well_evolution()
  plt.show()
