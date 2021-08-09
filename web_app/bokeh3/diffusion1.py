from bokeh.core.enums import SizingMode
import numpy as np
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, Toggle, Button, tools
from bokeh.transform import linear_cmap
from bokeh.plotting import figure


""" Note button types are default, primary, success, warning, danger, light """

# Parameters
n = 150             # number of particles
m = 630             # bin size
x_lim = 100         # (-x_lim, x_lim)
mu, sigma = 0, 5    # initial distribution
dx = 2*x_lim/(m-1)  # Gridstep
dt = 0.05           # Timestep
D = 1               # Diffusion coefficient 
F = D*dt/(dx**2)
first_time = True

def gaussian(x,sigma):
    return (1/np.sqrt(2*np.pi*sigma**2)) * np.exp( -(x-mu)**2 / (2*sigma**2) )

# Function source
x2 = np.linspace(-x_lim,x_lim,num=m)
y2 = gaussian(x2,sigma)
function_source = ColumnDataSource(data=dict(x=x2,y=y2))

def get_points(y,sigma=None):
    ymax = np.amax(y)
    if not sigma:
        sigma = 1/(ymax*np.sqrt(2*np.pi))
    targets = np.linspace(ymax,0,num=int(n/2),endpoint=False)
    arg = -2*sigma*np.log(sigma*np.sqrt(2*np.pi)*targets)
    x = np.concatenate((arg,-arg))
    if x.size != n and (n % 2) != 0:
        x = np.append(x,0)
    return x

# Particles source
x1 = get_points(y2)
cx = x_lim-np.abs(x1)
y1 = np.random.rand(n) / 2
X = [(i, x1[i]) for i in range(x1.size)]
particles_source = ColumnDataSource(data=dict(x=x1,y=y1,color=cx))
color_mapper = linear_cmap(field_name='color', palette='Blues8', low=x_lim, high=0, low_color='#084594', high_color='#f7fbff')

# Particle plot
p1 = figure(x_range=(-x_lim, x_lim), y_range=(-0.05,0.55), plot_height=150, plot_width=700, title='Classical Diffusion', tools='')
p1.circle(x='x',y='y', source=particles_source, size=12, color=color_mapper, alpha=0.75, line_alpha=0.0) # color='#037ffc'
p1.axis.visible = False
p1.toolbar.logo = None
p1.xgrid.visible = False
p1.ygrid.visible = False
#p1.title.text_font_size = '16pt'

# Function plots
p2 = figure(x_range=(-x_lim, x_lim), y_range=(0, 0.08), plot_height=300, plot_width=700, tools='')
p2.line(x='x', y='y', source=function_source, line_width=2)
p2.yaxis.visible = False
p2.toolbar.logo = None

# Initial distribution width slider
width_slider = Slider(start=0.5, end=50, value=sigma, step=0.5, title='Initial Distribution width')
def update_initial_width(attr,old,new):
    if first_time:
        sigma = width_slider.value
        y = gaussian(x2,sigma)
        function_source.data['y'] = y
        x = get_points(y,sigma)
        particles_source.data['x'] = x
        particles_source.data['color'] = x_lim-np.abs(x)
width_slider.on_change('value', update_initial_width)

# Diffusion constant slider
diff_slider = Slider(start=0.5, end=10, value=2, step=0.5, title='Diffusion Constant')
def update_diff(attr,old,new):
    global current_call
    x = diff_slider.value
    y = round(200/x)-10
    curdoc().remove_periodic_callback(current_call)
    current_call = curdoc().add_periodic_callback(callback, y) # 100 ms alternatively 
    return
diff_slider.on_change('value', update_diff)

# Evolve button
evolve_button = Toggle(label = '► Evolve', button_type='success')
def evolve_click(value):
    if evolve_button.active:
        evolve_button.label = '❚❚ Pause'
    else:
        evolve_button.label = '► Evolve'
    update_diff(1,2,3)
evolve_button.on_click(evolve_click)

# Reset button
reset_button = Button(label='Reset', button_type='primary')
def reset_click(value):
    evolve_button.label = '► Evolve'
    evolve_button.active = False
    global first_time
    if not first_time:
        first_time = True
        sigma = width_slider.value
        y = gaussian(x2,sigma)
        function_source.data['y'] = y
        x = get_points(y,sigma)
        particles_source.data['x'] = x
        particles_source.data['color'] = x_lim-np.abs(x)
    update_diff(1,2,3)
reset_button.on_click(reset_click)

def callback():
    if evolve_button.active:
        global first_time
        first_time = False
        
        y = function_source.data['y'] # current time step
        y_new = np.zeros(x2.size) # next time step
        for i in range(x2.size-1):
            y_new[i] = y[i] + F*(y[i-1] - 2*y[i] + y[i+1])
        function_source.data['y'] = y_new
        
        x = get_points(y_new)
        particles_source.data = dict(x=x,y=y1,color=x_lim-np.abs(x))

"""
Simulate for 100 ms (ide solve)
100 ms real / 100 ms evolve
Farg to ide solve

bug --> use closure, create d/dt function, assign a variable and pass it in
"""

# Set up layouts and add to document
inputs = column(row(evolve_button, reset_button, sizing_mode='scale_width', width=700),p1,p2,width_slider, diff_slider, width=700)
curdoc().add_root(inputs) # Need to adjust sizes just a bit more
curdoc().title = "Diffusion"
current_call = curdoc().add_periodic_callback(callback, 10) # 100 ms alternatively 