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
from bokeh.models.arrow_heads import TeeHead
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Toggle, Button, RadioButtonGroup
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
clock_source = ColumnDataSource(data=dict(x=[0, 0], y= [0, 1], theta=[0, -90]))

# Set up plot
plot_pdf = figure(plot_height=400, plot_width=520, title="Wavefunction pdf",
              tools="",
              x_range=[x_min, x_max])

plot_pdf.line('phi', 'pdf', source=source, line_width=3, line_alpha=0.6, line_color='black')
plot_pdf.toolbar.logo = None

plot_real = figure(plot_height=200, plot_width=270, title="Real part",
              tools="",
              x_range=[x_min, x_max])

plot_real.line('phi', 'real', source=source, line_width=3, line_alpha=0.6, line_color='red')
plot_real.toolbar.logo = None

plot_imag = figure(plot_height=200, plot_width=270, title="Imaginary part",
              tools="",
              x_range=[x_min, x_max])

plot_imag.line('phi', 'imag', source=source, line_width=3, line_alpha=0.6, line_color='orange')
plot_imag.toolbar.logo = None

plot_clock = figure(plot_height=100, plot_width=100, title="", tools="",  x_range=[-1, 1], y_range=[-1, 1])
plot_clock.circle([0], [0], size=60, color="navy", alpha=0.5)
plot_clock.line('x', 'y', source=clock_source, line_width=3, color="black")
plot_clock.toolbar.logo = None
plot_clock.xaxis.visible = False
plot_clock.yaxis.visible = False

# Set up widgets
measure = Button(label="Measure system", button_type="success")
evolve_button = Toggle(label = '► Evolve', button_type = "success")
regions = Slider(title="Number or regions", value=2, start=1, end=20, step=1)
reset_button = Button(label='Reset', button_type='primary')
radio_button_group = RadioButtonGroup(labels=['Classical', 'Quantum'], active=0)

def classic_measure(pdf, xrange, numregions):
    probability_table = []
    # get del x
    delx = xrange[1] - xrange[0]
        # for every region:
    for i in range(numregions):
        inds = [j for j in range(round(i*len(pdf)/numregions), round((i+1)*len(pdf)/numregions))]
        # create a  projection matrix x:
        x = np.zeros(pdf.shape)
        x[inds] = delx
        # find probability of flux being in that region by taking <phi^* | x | phi> and store in the probability table 
        prob = np.abs(np.dot(pdf, x))
        probability_table.append(prob)
    # Use multinomial RV to get the resul of throwing a weighted cube. Multinomial returns an array of size p.size where the entry in each index is the number of times
    # the cube landed on that face
    probability_table = np.array(probability_table)/np.sum(probability_table)   # normalize probabilities in case wavevctor isn't normalized
    print(probability_table)
    cube_throw = np.random.multinomial(1, probability_table)
    print(cube_throw)
    region_number = int((np.where(cube_throw ==1)[0][0]))  # for some odd reason numpy returns the array index s a float which needs to be converted to an int for indexing
    inds = [j for j in range(round(region_number*len(pdf)/numregions), round((region_number+1)*len(pdf)/numregions))]
    new_pdf = np.zeros(pdf.shape)
    new_pdf[inds] = 1  
    # normalize it
    new_pdf /= np.sum(np.absolute(new_pdf)*delx)
    return np.sqrt(new_pdf)



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
    clock_source.data = dict(x=[0, 0], y= [0, 1], theta=[0, -90])

reset_button.on_click(reset_click)

def measurement():
    if radio_button_group.active == 1:
        print("quantum measure")
        num_of_regions = regions.value
        wave_object = wavesource.data['wave']
        new_wave = wave_object.simple_measure_1d(num_of_regions)
        wavesource.data = dict(wave=new_wave)
        source.data['pdf'] = np.power(np.absolute(wavesource.data['wave']), 2)
        source.data['real'] = np.real(wavesource.data['wave'])
        source.data['imag'] = np.imag(wavesource.data['wave'])
    elif radio_button_group.active == 0:
        print("classical measure")
        num_of_regions = regions.value
        wave_object = wavesource.data['wave']
        new_wave = Wavevector(classic_measure(np.power(wave_object, 2), source.data['phi'], num_of_regions), wave_object.ranges)
        wavesource.data = dict(wave=new_wave)
        source.data['pdf'] = np.power(np.absolute(wavesource.data['wave']), 2)
        source.data['real'] = np.real(wavesource.data['wave'])
        source.data['imag'] = np.imag(wavesource.data['wave'])

measure.on_click(measurement)

def change_mode(value):
    if radio_button_group.active == 1:
        print("switched to quantum")
        source.data['pdf'] = np.power(np.absolute(wavesource.data['wave']), 2)
        source.data['real'] = np.real(wavesource.data['wave'])
        source.data['imag'] = np.imag(wavesource.data['wave'])
        #wave_object = Wavevector(wavesource.data['wave'].astype(complex), *dim_info)
        #wavesource.data = dict(wave=wave_object)
        return
    elif radio_button_group.active == 0:
        print("switched to classical")
        #wave_object = np.abs(wavesource.data['wave'])
        #wavesource.data = dict(wave=wave_object)
        source.data['pdf'] = np.power(np.absolute(wavesource.data['wave']), 2)
        source.data['real'] = np.sqrt(source.data['pdf'])
        #source.data['imag'] = np.imag(wavesource.data['wave'])
        source.data['imag'] = np.zeros(source.data['imag'].shape)
        return
radio_button_group.on_click(change_mode)


def callback():
    if evolve_button.active and radio_button_group.active == 1:
        start = time.time()
        time_t = time_source.data['time']
        print(type(wavesource.data['wave']))
        r = wavesource.data['wave'].evolve(lambda x: x-x, masses, (0, 0.1), frames = len(time_t), t_dep = False)
        for i in range(len(time_t)):
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
        new_theta = (clock_source.data['theta'][1]+360/100)%360
        new_x = np.cos(-new_theta*np.pi/180)
        new_y = np.sin(-new_theta*np.pi/180)
        clock_source.data = dict(x=[0, new_x], y=[0, new_y], theta=[0, new_theta])
        end = time.time()
        print("time lapsed", end-start)

    elif evolve_button.active:
        new_theta = (clock_source.data['theta'][1]+360/100)%360
        new_x = np.cos(-new_theta*np.pi/180)
        new_y = np.sin(-new_theta*np.pi/180)
        clock_source.data = dict(x=[0, new_x], y=[0, new_y], theta=[0, new_theta])
        


    


# Set up layouts and add to document
inputs = column(radio_button_group, row(measure, evolve_button, reset_button, width=900), regions, row(column(plot_imag, plot_real), plot_pdf, plot_clock), width=900)

curdoc().add_root(inputs)
curdoc().title = "Measurement"
curdoc().add_periodic_callback(callback, 100)