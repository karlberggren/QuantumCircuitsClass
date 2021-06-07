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
from bokeh.models import ColumnDataSource, Slider, Button, TextInput
from bokeh.plotting import figure, show

# Set up data
N = 200
C = 1
L = 1
x_range = 3
phi_0 = 0
Q_0 = 0
phi = np.linspace(-x_range, x_range, N)
energy = (phi - phi_0)**2/(2*L)
t_initial = 0
t_final = np.pi/16
time_t = np.linspace(t_initial, t_final, 7)
psi = np.exp(-(phi - phi_0)**2/(L*C))

source = ColumnDataSource(data=dict(phi=phi, energy=energy, psi=psi))
time_source = ColumnDataSource(data=dict(time=time_t))

# set up grid: each quadrant is actually goldern-ratio rectangle

# quadrant 1: energy vs. flux

plot_width = 500
φ = 1.61  # golden ratio
plot_height = int(plot_width / φ)

plot1 = figure(plot_height=plot_height, plot_width=plot_width, title="Energy vs. Flux",
               tools="crosshair,pan,reset,save,wheel_zoom",
               x_range=[-x_range, x_range], y_range=[-0.5, 1.5])

plot1.line('phi', 'energy', source=source, line_width=3, line_alpha=0.6)

# quadrant 2: L, C, Φ_EXT sliders
text = TextInput(title="title", value='LC potential')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
inductance = Slider(title="L", value=1.0, start=0.1, end=5.0, step=0.1)
capacitance = Slider(title="C", value=1.0, start=0.1, end=5.0)
quad2 = gridplot([[inductance,capacitance],[offset,text]], plot_width = int(plot_width/2), plot_height = int(plot_height/2))

# quadrant 3: PDF vs Φ
quad3 = figure(plot_height=plot_height,
               plot_width=plot_width, title="PDF vs. Flux",
               tools="crosshair,pan,reset,save,wheel_zoom",
               x_range=[-x_range, x_range], y_range=[-0.5, 1.5])
quad3.line('phi', 'psi', source=source, line_width=3, color="black", alpha=0.5)

# quadrant 4: ⟨Φ⟩(t=0) and σ_Φ(t=0) sliders, play/pause, reset buttons
offset = Slider(title="⟨Φ⟩(t=0)", value=0.0, start=-5.0, end=5.0, step=0.1)
σ = Slider(title="σ", value=0.1, start=0.1, end=5.0, step=0.1)
start_pause = Button(label = "Start/Pause", button_type="success")
reset = Button(label = "Reset", button_type = "success")
quad4 = gridplot([[start_pause, reset]])

# Set up widgets

# Set up callbacks
def update_title(attrname, old, new):
    plot1.title.text = text.value

text.on_change('value', update_title)

def update_data(attrname, old, new):
    # Get the current slider values
    L = inductance.value
    C = capacitance.value
    phi_0 = offset.value

    # Generate the new curve
    phi = np.linspace(-x_range, x_range, N)
    energy = (phi - phi_0)**2/(2*L)
    psi = np.exp(-(phi - phi_0)**2/np.sqrt(L*C))
    source.data = dict(phi=phi, energy=energy, psi=psi)

def callback():
    L = inductance.value
    C = capacitance.value
    phi_0 = offset.value
    time_t = time_source.data['time']
    for t in time_t:
        psi = np.exp(-(phi - phi_0 - np.sin(t*np.sqrt(L*C)))**2/np.sqrt(L*C))
        source.data['psi'] = psi
    t_initial = time_t[-1]
    t_final = time_t[-1]+np.pi/16
    time_t = np.linspace(t_initial, t_final, 7)
    time_source.data['time'] = time_t   

for w in [inductance, capacitance, offset]:
    w.on_change('value', update_data)

# Set up layouts and add to document
inputs = column(text, inductance, capacitance, offset)

grid = gridplot([[plot1,quad2],[quad3,quad4]], plot_width = plot_width, plot_height = plot_height)

show(grid)

curdoc().add_root(row(inputs, plot1, width=800))
curdoc().title = "Sliders"
curdoc().add_periodic_callback(callback, 150)
