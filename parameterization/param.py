''' 
Learning Goal: 

Understand how variation of circuit parameters influences energy-landscape of system evolution

Undertand that evolution of a quantum system can be affected by changing external parameters (e.g. external flux)

Circuit parameters such as source strengths, Present an interactive function explorer with slider widgets.

launch bokeh server locally with

bokeh serve --show parametrization.py

http://localhost:5006/parametrization
'''

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Button, TextInput, Circle, CDSView, BooleanFilter
from bokeh.plotting import figure, show

# this is probably kludgy, but the main code base is in "main", so need
# to go there first
import sys
sys.path.insert(0, '../main/')
from wavevector import Wavevector
from wavefunction import Wavefunction

# adding logging to help debugging
import logging

""" 
Time is hard to keep track of.  There are a number of time scales involved.

callback_period: period between callbacks in ms (typically 100 ms).

t: time in sim-time since last reset. at each reset, t should be reset to
   zero

update_time: not sure what this is
"""

# Set up sim and vis parametsr
callback_period = 100  #ms

# using this as a flag to check for reset.  Reset if zero.
update_time = callback_period

# Set up data
ħ = 1.05e-34
#FIXME
# Φo = 2.07e-15
Φo = 1

π = np.pi
N = 81
#FIXME
# C_scale, L_scale = 1e-15, 1e-11  # kludge because bokeh doesn't play nice with scientific notation
C_scale, L_scale = 0.1, 0.1

C_center, C_step, C_min, C_max  = 10*C_scale, 1*C_scale, 1*C_scale, 100*C_scale
L_center, L_step, L_min, L_max  = 10*L_scale, 1*L_scale, 1*L_scale, 100*L_scale
L, C = L_center, C_center

#FIXME x_range = 3*Φo
σ = np.sqrt(ħ/2*np.sqrt(L_center/C_center))
x_scale = σ
x_range = 6*x_scale
phi_ext = 0
phi_0 = 0
Q_0 = 0
phi = np.linspace(-x_range, x_range, N)
energy = (phi - phi_ext)**2/(2*L_center)

ω = 1/np.sqrt(L_center*C_center)
T = 2 * π / ω
logging.warning('Period is ' + str(T)) #FIXME

t = 0

dim_info = ((-x_range, x_range, N),)
masses = (C_center,)
#σ = 0.5

wv_o = Wavevector.from_wf(Wavefunction.init_gaussian((phi_0, σ)), *dim_info)
pdf = np.abs(wv_o)**2

δφ = 2*x_range / N
booleans = [True if (phi_0 - δφ) < x < (phi_0 + δφ) else False for x in phi]

source = ColumnDataSource(data=dict(phi=phi, energy=energy, pdf=pdf))
classical_view = CDSView(source=source, filters=[BooleanFilter(booleans)])

# set up grid: each quadrant is actually goldern-ratio rectangle

# quadrant 1: energy vs. flux

plot_width = 500
φ = 1.61  # golden ratio
plot_height = int(plot_width / φ)

quad1 = figure(plot_height=plot_height, plot_width=plot_width,
               title="Energy vs. Flux",
               tools="crosshair,pan,reset,save,wheel_zoom",
               x_range=[-x_range, x_range], y_range=[0, energy[-1]])

quad1.line('phi', 'energy', source=source, line_width=3, line_alpha=0.6)
quad1.circle('phi', 'energy', source=source, line_width=3, view=classical_view)

# quadrant 2: L, C, Φ_EXT sliders
init_offset = Slider(title="phi_0", value=0.0, start=-5.0, end=5.0, step=0.1)
offset = Slider(title="⟨Φ⟩_{EXT}", value=0.0, start=-5.0, end=5.0, step=0.1)
inductance = Slider(title="L [μH]", value=L_center/L_scale, start=L_min/L_scale, end=L_max/L_scale, step=L_step/L_scale, format = '0.0f')
capacitance = Slider(title="C [fF]", value=C_center/C_scale, start=C_min/C_scale, end=C_max/C_scale, step=C_step/C_scale, format = '0.0e')

quad2 = gridplot([[inductance,capacitance],[offset,init_offset],[None,None]], plot_width = int(plot_width/2), plot_height = int(plot_height/2))

# quadrant 3: PDF vs Φ
quad3 = figure(plot_height=plot_height,
               plot_width=plot_width, title="PDF vs. Flux",
               tools="crosshair,pan,reset,save,wheel_zoom",
               x_range=[-x_range, x_range], y_range=[0, 1/(2*σ)])
#               x_range=[-x_range, x_range], y_range=[0, 1]) FIXME
quad3.line('phi', 'pdf', source=source, line_width=3, color="black", alpha=0.5)

# quadrant 4: ⟨Φ⟩(t=0) and σ_Φ(t=0) sliders, play/pause, reset buttons
start_pause = Button(label = '⏸')
#reset = Button(label = "Reset", button_type = "success")
reset = Button(label = "Reset")
quad4 = gridplot([[start_pause, reset]])

# Set up widgets

def update_data(attrname, old, new):
    global t, C, L, wv_o, phi_ext, phi_0, T
    # Get the current slider values
    L = inductance.value*L_scale
    C = capacitance.value*C_scale
    ω = 1/np.sqrt(L*C)
    T = 2 * π / ω
    logging.warning('Period is ' + str(T)) #FIXME
    phi_ext = offset.value*x_scale
    phi_0 = init_offset.value*x_scale
    
    # Generate the new curve
    energy = (phi - phi_ext)**2/(2*L)

    source.data = dict(phi=phi, energy=energy, pdf=pdf)
    booleans = [True if (phi_0 - phi_ext*np.cos(ω*t) - δφ/2) < x < (phi_0 - phi_ext*np.cos(ω*t) + δφ/2) else False for x in phi]
    classical_view.filters[0] = BooleanFilter(booleans)
    #source.view = CDSView(source=source, filters=[BooleanFilter(booleans)])


def callback():
    global t, C, L, wv_o, phi_ext, phi_0, pdf, T
    # move forward by callback_period
    L = inductance.value*L_scale
    C = capacitance.value*C_scale
    phi_ext = offset.value*x_scale
    phi_0 = init_offset.value*x_scale

    #time_t = time_source.data['time']
    def V(Φ):
        return (Φ - phi_ext)**2/(2*L)/ħ  # FIXME remove ħ
        #return Φ - Φ  # kludge to return array of 0s FIXME

    """ if we haven't recently had a reset """
    if update_time :
        masses = (C,)
        #masses = (ħ,) FIXME

        # note, need frames = 2, else just returns the initial frame
        r = wv_o.evolve(V,
                        masses,
                        (0, T/100),
                        frames = 2,
                        t_dep = False)
        pdf = np.abs(r.y.T[-1])**2
        wv_o = Wavevector(r.y.T[-1], wv_o.ranges)  # make a copy

    #pdf = 1/(2*π*σ**2)**(0.25)*np.exp(-(phi - phi_0 - phi_ext*np.cos(ω*t))**2/(4*σ**2))
    source.data['pdf'] = pdf
    # CLASSICAL POINT
    """ 
    I now need to filter out any of the points that are not the classical
    point 
    """
    
    booleans = [True if (phi_0 - phi_ext*np.cos(ω*t) - δφ/2) < x < (phi_0 - phi_ext*np.cos(ω*t) + δφ/2) else False for x in phi]
    classical_view.filters[0] = BooleanFilter(booleans)

    t += update_time/1000*T  # sim time passed

# check for any slider changes, and if so, run update_data accordingly

for w in [offset, inductance, capacitance, init_offset]:
    w.on_change('value', update_data)

def update_start_pause():
    global update_time, start_pause
    if update_time > 0:
        update_time = 0
        start_pause.label = '▶'
    else:
        update_time = callback_period
        start_pause.label = '⏸'

def update_reset():
    global t
    global wv_o, phi_0, phi_ext, L, C
    global pdf
    L = inductance.value*L_scale
    C = capacitance.value*C_scale
    phi_ext = offset.value*x_scale
    phi_0 = init_offset.value*x_scale
    σ = np.sqrt(ħ/2*np.sqrt(L/C))

    energy = (phi - phi_ext)**2/(2*L)

    wv_o = Wavevector.from_wf(Wavefunction.init_gaussian((phi_ext, σ)), *dim_info)
    pdf = np.abs(wv_o)**2
    source.data = dict(phi=phi, energy=energy, pdf=pdf)
    # FIXME bug here, where energy data isn't being reset properly
    t = 0  # restart sim time

start_pause.on_click(update_start_pause)
reset.on_click(update_reset)

# Set up layouts and add to document

grid = gridplot([[quad1,quad2],[quad3,quad4]],
                plot_width = plot_width,
                plot_height = plot_height)

#show(grid)

curdoc().add_root(grid)
curdoc().title = "Sliders"
curdoc().add_periodic_callback(callback, callback_period)
