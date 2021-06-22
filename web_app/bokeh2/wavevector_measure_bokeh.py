''' Present an interactive function explorer with slider widgets.
Scrub the sliders to change the properties of the ``sin`` curve, or
type into the title text box to update the title of the plot.
Use the ``bokeh serve`` command to run the example by executing:
    bokeh serve sliders.py
at your command prompt. Then navigate to the URL
    http://localhost:5006/sliders
in your browser.
'''
from bokeh.core.property.color import Color
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Toggle, Button
from bokeh.plotting import figure
from utils.wavevector import Wavevector
from utils.wavefunction import Wavefunction
import time

# global constants
π = np.pi
oo = np.inf
ħ = 1

# initiate wavevector
x_min, x_max, N = -10, 10, 200
dim_info = ((x_min, x_max, N),)
masses = (ħ,)
#wv_o = Wavevector.from_wf((1+1j)/np.sqrt(2)*Wavefunction.init_gaussian((0,1)), *dim_info)
wv_o = Wavevector.from_wf(Wavefunction.init_gaussian((0,1)), *dim_info)


phi = np.linspace(x_min, x_max, N, N)
t_initial = 0
t_final = np.pi/16
time_t = np.linspace(t_initial, t_final, 2)


wavesource = ColumnDataSource(data=dict(wave= wv_o))
pdf = np.power(np.absolute(wavesource.data['wave']), 2)
real = np.real(wavesource.data['wave'])
imag = np.imag(wavesource.data['wave'])
source = ColumnDataSource(data=dict(phi=phi, pdf=pdf, real=real, imag=imag))
time_source = ColumnDataSource(data=dict(time=time_t))

# Set up plot
plot_pdf = figure(plot_height=400, plot_width=600, title="Wavefunction pdf",
              tools="",
              x_range=[x_min, x_max])

plot_pdf.line('phi', 'pdf', source=source, line_width=3, line_alpha=0.6, line_color='black')
plot_pdf.toolbar.logo = None

plot_real = figure(plot_height=200, plot_width=300, title="Real part",
              tools="",
              x_range=[x_min, x_max])

plot_real.line('phi', 'real', source=source, line_width=3, line_alpha=0.6, line_color='red')
plot_real.toolbar.logo = None

plot_imag = figure(plot_height=200, plot_width=300, title="Imaginary part",
              tools="",
              x_range=[x_min, x_max])

plot_imag.line('phi', 'imag', source=source, line_width=3, line_alpha=0.6, line_color='orange')
plot_imag.toolbar.logo = None

# Set up widgets
measure = Button(label="Measure system", button_type="success")
evolve_button = Toggle(label = '► Evolve', button_type = "success")
regions = Slider(title="Number or regions", value=2, start=1, end=20, step=1)
reset_button = Button(label='Reset', button_type='primary')

def evolve_click(value):
    if evolve_button.active:
        evolve_button.label = '❚❚ Pause'
    else:
        evolve_button.label = '► Evolve'
evolve_button.on_click(evolve_click)

def reset_click(value):
    evolve_button.label = '► Evolve'
    evolve_button.active = False
    wavesource.data['wave'] = wv_o
    source.data['pdf'] = np.power(np.absolute(wavesource.data['wave']), 2)
    source.data['real'] = np.real(wavesource.data['wave'])
    source.data['imag'] = np.imag(wavesource.data['wave'])
    time_source.data['time'] = np.linspace(t_initial, t_final, 2)

reset_button.on_click(reset_click)

def quantum_measurement():
    print("quantum measure")
    num_of_regions = regions.value
    wave_object = wavesource.data['wave']
    new_wave = wave_object.simple_measure_1d(num_of_regions)
    wavesource.data = dict(wave=new_wave)
    source.data['pdf'] = np.power(np.absolute(wavesource.data['wave']), 2)
    source.data['real'] = np.real(wavesource.data['wave'])
    source.data['imag'] = np.imag(wavesource.data['wave'])


measure.on_click(quantum_measurement)

def callback():
    if evolve_button.active:
        start = time.time()
        time_t = time_source.data['time']
        r = wavesource.data['wave'].evolve(lambda x: x-x, masses, (0, 0.1), frames = len(time_t), t_dep = False)
        for i in range(len(time_t)):
            print(i)
            wave = r.y.T[i, :]
            source.data['pdf'] = np.power(np.absolute(wave), 2)
            source.data['real'] = np.real(wave)
            source.data['imag'] = np.imag(wave)

        t_initial = time_t[-1]
        t_final = time_t[-1]+np.pi/16
        time_t = np.linspace(t_initial, t_final, 2)
        time_source.data['time'] = time_t   
        new_wave = Wavevector(r.y.T[-1], wavesource.data['wave'].ranges) 
        wavesource.data = dict(wave = new_wave)
        end = time.time()
        print("time lapsed", end-start)
    


# Set up layouts and add to document
inputs = column(row(measure, evolve_button, reset_button), regions, row(column(plot_imag, plot_real), plot_pdf))

curdoc().add_root(row(inputs, width=800))
curdoc().title = "Measurement"
curdoc().add_periodic_callback(callback, 100)