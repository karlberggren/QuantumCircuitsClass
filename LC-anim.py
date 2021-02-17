import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import rc
import scipy.integrate as integrate
from scipy.integrate import solve_ivp
from matplotlib.patches import Circle
from matplotlib.widgets import Slider, Button, CheckButtons
import time
import copy

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

                     In this analog, a statmeent like p = m v (the integral of F = m A)
                     is analogous to the statement Q = C V (the integral of i = C ∂_t).
                     Similarly, the potential energy stored in a spring (V(x) = ½ k x²)
                     maps neatly onto the potential energy stored in an inductor
                     E_L = ½ L i².

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
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)  # make room for widgets

# create widget to change x limits
axcolor = 'lightgoldenrodyellow'
ax_xlim = plt.axes([0.25, 0.1, 0.64, 0.03], facecolor = axcolor)
xlims = Slider(ax_xlim, 'x limit', 0.1, 5, valinit = 1.5, valstep = 0.2)

# widget to change capacitance
ax_cap = plt.axes([0.25, 0.15, 0.64, 0.03], facecolor = axcolor)
cap = Slider(ax_cap, 'capacitance', 0.2, 3, valinit = 1, valstep = 0.2)

# widget to change inductance
ax_ind = plt.axes([0.25, 0.2, 0.64, 0.03], facecolor = axcolor)
ind = Slider(ax_ind, 'inductance', 0.2, 3, valinit = 1, valstep = 0.2)

ccs = []
for starting_pos in [0.25, 0.5, 0.75, 1]:
    ccs.append(Classical_circuit(starting_pos, 0, 1, LC_V, LC_dVdx))

colors = ['yo', 'bo', 'go', 'ro']
visibility = [True, True, True, True]

Δt = 50  # in ms
def anim_func(i):
    for cc in ccs:  # update mass when slider moves
        cc.m_eff = cap.val
        cc.dVdx = lambda t, i:  ind.val * i
        cc.V = lambda t, i: 1/2 * ind.val * i**2
    t_o = i * Δt / 1000
    t_f = (i + 1) * Δt / 1000
    ax.clear()
    ax.set_xlim(-xlims.val, xlims.val)
    xs = np.linspace(-xlims.val, xlims.val)
    V = ccs[0].V
    ax.set_ylim(V(0,0), V(0, xlims.val))
    ax.plot(xs, V(0, xs), color='tab:blue')  # potential
    for n, cc in enumerate(ccs):
        _, xs, ps = cc.sim((t_o, t_f))
        cc.x_o, cc.p_o = xs[-1], ps[-1]
        x = cc.x_o
        ax.plot(x, V(t_f, x), colors[n], visible = visibility[n])

    
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
