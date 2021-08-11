import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Toggle, Button, RadioButtonGroup, ImageURL
from bokeh.plotting import figure
from scipy.optimize import curve_fit

# Simulation parameters
x_lim = 100                             # (-x_lim, x_lim) bounds
x = np.linspace(-x_lim,x_lim,num=630)   # grid for simulation of diffusion num is number of points
dx = x[1] - x[0]                        # gridstep
t = 0                                   # initial time
dt = 0.05                               # timestep
F = dt/(dx**2)
first_time = True
switched = False

# Initial conditions for gaussian distribution
mu, sigma = 0, 5
L = 10

# Convenience functions
def gaussian(x,sigma):
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp( -(x-mu)**2 / (2*sigma**2) )

def psi(x,t,L,sigma):
    a = 1/(2*sigma**2)
    b = 1+(2j*a*t/L)
    return (2*a/np.pi)**(1/4) * (1/np.sqrt(b)) * np.exp((-a*x**2)/(b))

def fity0(x,sigma):
    y = psi(x,0,L,sigma)
    y2 = np.sqrt(y.real**2+y.imag**2)
    return y2.real
    
# Function source
y = np.zeros(x.size, dtype='complex')
y = y + psi(x,0,L,sigma)
y2 = np.sqrt(y.real**2+y.imag**2)
function_source = ColumnDataSource(data=dict(x=x,y=y2.real))
real_source = ColumnDataSource(data=dict(x=x,y=y.real))
imag_source = ColumnDataSource(data=dict(x=x,y=y.imag))

# Real figure
p_real = figure(x_range=(-x_lim, x_lim), y_range=(-0.35, 0.35), plot_width=220, plot_height=300, title='Real Component Re{Ψ}', tools='')
p_real.line(x='x', y='y', source=real_source, line_width=2)
p_real.xaxis.axis_label = 'Flux [Wb]'
p_real.yaxis.visible = False
p_real.toolbar.logo = None
p_real.xgrid.visible = False
p_real.ygrid.visible = False
p_real.xaxis.major_label_text_color = "white"

# Circuit image
p_image = figure(x_range=(0,200), y_range=(0,200), plot_width=230, plot_height=300, tools='')
url = "https://bokeh4-dot-quantum-explorations.uk.r.appspot.com/circuit.png"
#url = 'circuit.png'
source = ColumnDataSource(data=dict(url=[url]))
image = ImageURL(url=dict(value=url), x=5, y=0, w=200, h=200, anchor="bottom_left")
p_image.add_glyph(source,image)
p_image.axis.visible = False
p_image.toolbar.logo = None
p_image.xgrid.visible = False
p_image.ygrid.visible = False

# Imaginary figure 
p_imag = figure(x_range=(-x_lim, x_lim), y_range=(-0.35, 0.35), plot_width=220, plot_height=300, title='Imaginary Component Im{Ψ}', tools='')
p_imag.line(x='x', y='y', source=imag_source, line_width=2)
p_imag.xaxis.axis_label = 'Flux [Wb]'
p_imag.yaxis.visible = False
p_imag.toolbar.logo = None
p_imag.xgrid.visible = False
p_imag.ygrid.visible = False
p_imag.xaxis.major_label_text_color = "white"

# Probability density figure
p = figure(x_range=(-x_lim, x_lim), y_range=(0, 0.5), plot_width=700, plot_height=300, title='Probability Density Function Ψ*Ψ', tools='')
p.line(x='x', y='y', source=function_source, line_width=2)
p.xaxis.axis_label = 'Flux [Wb]'
p.yaxis.visible = False
p.toolbar.logo = None
p.xgrid.visible = False
p.ygrid.visible = False
p.xaxis.major_label_text_color = "white"

# Slider update
def update_slider(attr,old,new):
    if first_time:
        L = inductance_slider.value
        sigma = width_slider.value
        if radio_button_group.active == 1:
            y = np.zeros(x.size, dtype='complex')
            y = y + psi(x,0,L,sigma)
            y2 = np.sqrt(y.real**2+y.imag**2)
            function_source.data['y'] = y2.real
            real_source.data['y'] = y.real
            imag_source.data['y'] = y.imag
        elif radio_button_group.active == 0:
            y = psi(x,0,L,sigma)
            y2 = y*y
            function_source.data['y'] = y2
            real_source.data['y'] = y.real
            imag_source.data['y'] = np.zeros(x.size)

# Inductance slider
inductance_slider = Slider(start=0.5, end=50, value=L, step=0.5, title='Inductance [nH]')
inductance_slider.on_change('value', update_slider)

# Initial distribution width slider
width_slider = Slider(start=0.5, end=50, value=sigma, step=0.5, title='Initial Distribution width [Wb]')
width_slider.on_change('value', update_slider)

# Evolve button
evolve_button = Toggle(label = '► Evolve', button_type='success')
def evolve_click(value):
    if evolve_button.active:
        evolve_button.label = '❚❚ Pause'
    else:
        evolve_button.label = '► Evolve'
evolve_button.on_click(evolve_click)

# Reset button
reset_button = Button(label='Reset', button_type='primary')
def reset_click(value):
    evolve_button.label = '► Evolve'
    evolve_button.active = False
    global first_time
    global t 
    if not first_time:
        first_time = True
        if radio_button_group.active == 1:
            t = 0
            y = np.zeros(x.size, dtype='complex')
            for i in range(x.size):
                y[i] += psi(x[i],0,inductance_slider.value,width_slider.value)
            y2 = np.sqrt(y.real**2+y.imag**2)
            function_source.data['y'] = y2
            real_source.data['y'] = y.real
            imag_source.data['y'] = y.imag
            return
        elif radio_button_group.active == 0:
            # reset original distribution
            y = np.zeros(x.size, dtype='complex')
            for i in range(x.size):
                y[i] += psi(x[i],0,inductance_slider.value,width_slider.value)
            y2 = np.sqrt(y.real**2+y.imag**2)
            function_source.data['y'] = y2
            real_source.data['y'] = y2
            imag_source.data['y'] = np.zeros(x.size)
            return
reset_button.on_click(reset_click)

radio_button_group = RadioButtonGroup(labels=['Diffusion', 'Schrodinger'], active=1)
def change_mode(value):
    if radio_button_group.active == 1:
        global sigma
        global t
        params = curve_fit(fity0,x[1:-1],function_source.data['y'][1:-1])
        sigma = params[0][0]
        t = 0
        y = np.zeros(x.size, dtype='complex')
        for i in range(x.size):
            y[i] += psi(x[i],0,inductance_slider.value,sigma)
        y2 = np.sqrt(y.real**2+y.imag**2)
        function_source.data['y'] = y2
        real_source.data['y'] = y.real
        imag_source.data['y'] = y.imag
        return
    elif radio_button_group.active == 0:
        real_source.data['y'] = function_source.data['y']
        imag_source.data['y'] = np.zeros(x.size)
        return
radio_button_group.on_click(change_mode)

def callback():
    global t
    global first_time
    if evolve_button.active:
        first_time = False
        L = inductance_slider.value
        sigma = width_slider.value
        if radio_button_group.active == 1:
            y = np.zeros(x.size, dtype='complex')
            for i in range(x.size):
                y[i] += psi(x[i],t,L,sigma)
            y2 = np.sqrt(y.real**2+y.imag**2)
            function_source.data['y'] = y2
            real_source.data['y'] = y.real
            imag_source.data['y'] = y.imag
            t += dt*100
        elif radio_button_group.active == 0:
            y = function_source.data['y']
            y_new = np.zeros(x.size)
            for i in range(x.size-1):
                y_new[i] = y[i] + F*(y[i-1] - 2*y[i] + y[i+1])
            function_source.data['y'] = y_new
            real_source.data['y'] = y_new
            imag_source.data['y'] = np.zeros(x.size)

""" Add unicode characters and infinte capacitance label """

# Set up layouts and add to document
inputs = column(row(evolve_button,radio_button_group,reset_button, width=700),column(width_slider,inductance_slider, width=700),row(p_imag,p_image,p_real),p)
curdoc().add_root(inputs)
curdoc().title = "Diffusion"
curdoc().add_periodic_callback(callback, 10)