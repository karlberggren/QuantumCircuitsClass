''' 
Learning Goal: 

Understand how variation of circuit parameters influences energy-landscape of system evolution

Undertand that evolution of a quantum system can be affected by changing external parameters (e.g. external flux)

Circuit parameters such as source strengths, Present an interactive function explorer with slider widgets.

http://localhost:5006/sliders
'''

import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row, gridplot
from bokeh.models import ColumnDataSource, Slider, Button, TextInput, Circle, CDSView, BooleanFilter
from bokeh.plotting import figure, show

from wavevector import Wavevector
from wavefunction import Wavefunction

# Set up sim and vis parametsr
callback_period = 100  #ms
update_time = callback_period

# Set up data
ħ = 1.05e-34
Φo = 2.07e-15
π = np.pi
N = 101
C_center = 10e-15
C_step = 1e-15
C_min = 1e-15
C_max = 100e-15
C = C_center
L_center = 10e-6
L_step = 1e-6
L_min = 1e-6
L_max = 100e-6
L = L_center
x_range = 3*Φo
phi_0 = 0
Q_0 = 0
phi = np.linspace(-x_range, x_range,N)
energy = (phi - phi_0)**2/(2*L_center)

ω = 1/np.sqrt(L_center*C_center)
T = 2 * π / ω
t = 0

dim_info = ((-x_range, x_range, N),)
masses = (C_center,)
σ = np.sqrt(ħ/2*np.sqrt(L_center/C_center))
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
inductance = Slider(title="L [μH]", value=L_center*1e6, start=L_min*1e6, end=L_max*1e6, step=L_step*1e6, format = '0.0f')
capacitance = Slider(title="C [fF]", value=C_center*1e15, start=C_min*1e15, end=C_max*1e15, step=C_step*1e15, format = '0.0f')

quad2 = gridplot([[inductance,capacitance],[offset,init_offset],[None,None]], plot_width = int(plot_width/2), plot_height = int(plot_height/2))

# quadrant 3: PDF vs Φ
quad3 = figure(plot_height=plot_height,
               plot_width=plot_width, title="PDF vs. Flux",
               tools="crosshair,pan,reset,save,wheel_zoom",
               x_range=[-x_range, x_range], y_range=[0, 1e15])
quad3.line('phi', 'pdf', source=source, line_width=3, color="black", alpha=0.5)

# quadrant 4: ⟨Φ⟩(t=0) and σ_Φ(t=0) sliders, play/pause, reset buttons
start_pause = Button(label = '⏸')
#reset = Button(label = "Reset", button_type = "success")
reset = Button(label = "Reset")
quad4 = gridplot([[start_pause, reset]])

# Set up widgets
def update_data(attrname, old, new):
    global t, C, L, wv_o, phi_ext, phi_0
    # Get the current slider values
    L = inductance.value/1e6
    C = capacitance.value/1e15
    phi_ext = offset.value*Φo
    phi_0 = init_offset.value*Φo
    
    # Generate the new curve
    energy = (phi - phi_ext)**2/(2*L)

    source.data = dict(phi=phi, energy=energy, pdf=pdf)
    booleans = [True if (phi_0 - phi_ext*np.cos(ω*t) - δφ/2) < x < (phi_0 - phi_ext*np.cos(ω*t) + δφ/2) else False for x in phi]
    classical_view.filters[0] = BooleanFilter(booleans)
    #source.view = CDSView(source=source, filters=[BooleanFilter(booleans)])


def callback():
    global t, C, L, wv_o, phi_ext, phi_0, pdf
    # move forward by callback_period
    L = inductance.value/1e6
    C = capacitance.value/1e15
    phi_ext = offset.value*Φo
    phi_0 = init_offset.value*Φo

    #time_t = time_source.data['time']
    def V(Φ):
        return (Φ - phi_ext)**2/(2*L)
    if update_time :
        masses = (C,)
        wv_o.evolve(V, masses, (0, update_time/1000*T*1e-4), frames = 1, t_dep = False)
        pdf = np.abs(wv_o)**2
    #pdf = 1/(2*π*σ**2)**(0.25)*np.exp(-(phi - phi_0 - phi_ext*np.cos(ω*t))**2/(4*σ**2))
    source.data['pdf'] = pdf
    booleans = [True if (phi_0 - phi_ext*np.cos(ω*t) - δφ/2) < x < (phi_0 - phi_ext*np.cos(ω*t) + δφ/2) else False for x in phi]
    classical_view.filters[0] = BooleanFilter(booleans)
    t += update_time/1000

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
    L = inductance.value/1e6
    C = capacitance.value/1e15
    phi_ext = offset.value*Φo
    phi_0 = init_offset.value*Φo
    σ = np.sqrt(ħ/2*np.sqrt(L/C))
    energy = (phi - phi_ext)**2/(2*L)

    wv_o = Wavevector.from_wf(Wavefunction.init_gaussian((phi_ext, σ)), *dim_info)
    pdf = np.abs(wv_o)**2
    source.data = dict(phi=phi, energy=energy, pdf=pdf)
    t = 0

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
 
