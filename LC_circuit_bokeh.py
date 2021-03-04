''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput
from bokeh.plotting import figure

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
t_final = np.pi/8
time_t = np.linspace(t_initial, t_final, 15)
psi = np.exp(-(phi - phi_0)**2/(L*C))

source = ColumnDataSource(data=dict(phi=phi, energy=energy, psi=psi))
time_source = ColumnDataSource(data=dict(time=time_t))

# Set up plot
plot = figure(plot_height=400, plot_width=400, title="LC potential",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[-x_range, x_range], y_range=[-0.5, 1.5])

plot.line('phi', 'energy', source=source, line_width=3, line_alpha=0.6)
plot.line('phi', 'psi', source=source, line_width=3, color="black", alpha=0.5)


# Set up widgets
text = TextInput(title="title", value='LC potential')
offset = Slider(title="offset", value=0.0, start=-5.0, end=5.0, step=0.1)
inductance = Slider(title="L", value=1.0, start=0.1, end=5.0, step=0.1)
capacitance = Slider(title="C", value=1.0, start=0.1, end=5.0)


# Set up callbacks
def update_title(attrname, old, new):
    plot.title.text = text.value

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
    t_final = time_t[-1]+np.pi/8
    time_t = np.linspace(t_initial, t_final, 15)
    time_source.data['time'] = time_t   

for w in [inductance, capacitance, offset]:
    w.on_change('value', update_data)


# Set up layouts and add to document
inputs = column(text, inductance, capacitance, offset)

curdoc().add_root(row(inputs, plot, width=800))
curdoc().title = "Sliders"
curdoc().add_periodic_callback(callback, 100)