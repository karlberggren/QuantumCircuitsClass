import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import rc
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from scipy.stats import norm
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, CheckButtons

SHOW_TEST_PLOTS = False
ħ = 1  # h = 6.63e-34 J s or 6.58e-16 eV s
       # ħ = h / 2π = 1.05 e -34
π = np.pi
Φₒ = 1  # Φₒ = 2.07 e -15 in reality.
ⅉ = 1j
plt.ion()

class Classical_circuit(object):
    """
    Class for simple one-dimensional (i.e. 2nd order) classical circuit simulation.
​
    >>> L = 1
    >>> LC_V = lambda t, i: 1/2 * L * i**2
    >>> LC_dVdx = lambda t, i: - L * i
    >>> cc = Classical_circuit(0, 1, 1, LC_V, LC_dVdx)
    >>> print(cc.sim((0,1)))
    (array([0.00000000e+00, 9.99000999e-04, 1.09890110e-02, 1.10889111e-01,
           7.60643578e-01, 1.00000000e+00]), array([0.00000000e+00, 9.99001165e-04, 1.09892322e-02, 1.11116507e-01,
           8.36136051e-01, 1.17519805e+00]), array([1.        , 1.0000005 , 1.00006038, 1.0061545 , 1.30352833,
           1.54309856]))
    """

    def __init__(self, x_o, p_o, m_eff, V, dVdx, analog = "admittance"):
        """
        x_o, p_o, m_eff::parameters that map onto charge, flux, capacitance and indcutance
                     as specified in "analog_mapping".  The meaning of x, p, m_eff, V
                     can be understood differently depending on the setting of "analog".
            
                     In the more common "admittance" or "firestone" analog, 
                     current maps to force and voltage maps to velocity (or in the state
                     language, charge maps to momentum, and flux maps to position).
​
                     In this analog, a statmeent like p = m v (the integral of F = m A)
                     is analogous to the statement Q = C V (the integral of i = C ∂_t).
                     Similarly, the potential energy stored in a spring (V(x) = ½ k x²)
                     maps neatly onto the potential energy stored in an inductor
                     E_L = ½ L i².
​
        V::potential function of the form V(t,x)
        dVdx::negative of force function of the form dV(t,x)/dx
    
        analog::text indicating what analogy is being used, either "impedance" or
            "admittance" where "admittance" is the Firestone analogy, and is 
            the default value.
        """
        analog_mapping = {"impedance": ("charge","flux","capacitance","inductance"),
                          "admittance": ("flux","charge","inductance","capacitance")
        }

        self.x_o = x_o
        self.p_o = p_o
        self.m_eff = m_eff
        self.V = V
        self.dVdx = dVdx

        
    def sim(self, times):
        """
        sim: run simulation
        params: dictionary of parameters
        times: tuple of start and end times (t_o, t_end)
        """

        def dvdt(t: "time", v: "Vector (x,p)"):

            """ helper function for solver that gives time derivative of state.
            Note 2nd parameter is in form of a tuple as shown. """
            x, p = v
            return (p/self.m_eff, -self.dVdx(t,x))

        r = solve_ivp(dvdt, times, (self.x_o, self.p_o), t_eval = None)
        return r.t, r.y[0], r.y[1]

L = 1
LC_V = lambda t, i: 1/2 * L * i**2
LC_dVdx = lambda t, i:  L * i
cc = Classical_circuit(0, 1, 1, LC_V, LC_dVdx)
if SHOW_TEST_PLOTS:
    fig, ax = plt.subplots()
    ax.plot(*cc.sim((0,10)))
    fig.show()

L = 1
LC_V = lambda t, i: 1/2 * L * i**2
LC_dVdx = lambda t, i:  L * i
fig, ax = plt.subplots(nrows=2,ncols=1,sharex=True,gridspec_kw={'height_ratios': [1, 4]})
plt.subplots_adjust(left=0.25, bottom=0.3)  # make room for widgets

axcolor = 'white' # 'lightgoldenrodyellow'
slidercolor = 'grey'

# create widget to change x limits
ax_xlim = plt.axes([0.25, 0.12, 0.64, 0.01], facecolor = axcolor)
xlims = Slider(ax_xlim, 'x limit', 0.1, 5, valinit = 1.5, valstep = 0.2, color=slidercolor)

# widget to change capacitance
ax_cap = plt.axes([0.25, 0.16, 0.64, 0.01], facecolor = axcolor)
cap = Slider(ax_cap, 'capacitance', 0.2, 3, valinit = 1, valstep = 0.2, color=slidercolor)

# widget to change inductance
ax_ind = plt.axes([0.25, 0.2, 0.64, 0.01], facecolor = axcolor)
ind = Slider(ax_ind, 'inductance', 0.2, 3, valinit = 1, valstep = 0.2, color=slidercolor)

# widget to change mu
ax_mu = plt.axes([0.25, 0.08, 0.64, 0.01], facecolor = axcolor)
mu = Slider(ax_mu, 'x_0', -5, 5, valinit = 0, valstep = 0.1, color=slidercolor)

# widget to change sigma
ax_sigma = plt.axes([0.25, 0.04, 0.64, 0.01], facecolor = axcolor)
sigma = Slider(ax_sigma, 'σ', 0.05, 1, valinit = 0.5, valstep = 0.05, color=slidercolor)


# add a pause button
pause = True
started = False
ax_pause = plt.axes([0.8, 0.825, 0.09, 0.04])
pause_button = Button(ax_pause, 'Start', color=axcolor, hovercolor='lightblue')
pause_dict = {False: "Pause", True: "Resume"}

def pause_event(event):
    global pause
    global started
    
    if not started:
        # Build points distribution
        g_points = np.random.normal(mu.val,sigma.val,size)
        for starting_pos in g_points:
            ccs.append(Classical_circuit(starting_pos, 0, 1, LC_V, LC_dVdx))  
        # Hide x_0 and σ sliders 
        started = True
        ax_mu.set_visible(False)
        ax_sigma.set_visible(False)

    pause ^= True
    pause_button.label.set_text(pause_dict[pause])

pause_button.on_clicked(pause_event)

ccs = []
size = 50 # number of points in distribution
Δt = 50  # in ms
t_o = 0

def anim_func(i):
    global t_o
    
    if not started:
        ax[0].clear()
        ax[0].set_ylim([0, 10])
        t_gauss = np.linspace(-xlims.val,xlims.val,num=size)
        gauss = norm.pdf(t_gauss,mu.val,sigma.val)
        ax[0].plot(t_gauss, gauss, linewidth=2, color='r')
        return    
    if pause:
        return
    
    for cc in ccs:  # update mass when slider moves
        cc.m_eff = cap.val
        cc.dVdx = lambda t, i:  ind.val * i
        cc.V = lambda t, i: 1/2 * ind.val * i**2
    t_f = t_o + Δt / 1000
    ax[1].clear()
    ax[1].set_xlim(-xlims.val, xlims.val)
    xs = np.linspace(-xlims.val, xlims.val)
    V = ccs[0].V
    ax[1].set_ylim(V(0,0), V(0, xlims.val))
    ax[1].plot(xs, V(0, xs), color='tab:blue')  # potential
    for n, cc in enumerate(ccs):
        _, xs, ps = cc.sim((t_o, t_f))
        cc.x_o, cc.p_o = xs[-1], ps[-1]
        x = cc.x_o
        ax[1].plot(x, V(t_f, x), 'bo', visible = True, alpha=0.2)
    t_o += Δt/1000

    # For histogram
    ax[0].clear()
    ax[0].set_xlim(-xlims.val, xlims.val)
    ax[0].set_ylim([0, 10])
    data = []
    for cc in ccs:
        data.append(cc.x_o)
    count, bins, ignored = ax[0].hist(data, bins=12, density=True, stacked=True)
    new_mu, new_sigma = norm.fit(data)
    t_gauss = np.linspace(-xlims.val,xlims.val,num=size)
    gauss = norm.pdf(t_gauss,new_mu,new_sigma)
    ax[0].plot(t_gauss, gauss, linewidth=2, color='r')
    
    
ani = FuncAnimation(fig, anim_func, interval = Δt)
fig.show()

def LC_pot(t, φ, params):
  """ LC_pot: potential for an LC circuit """
  return φ**2/2/params["L"](t)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    # params["potential"] = LC_pot

# anim(params)
# plt.show()

print("this is running")
