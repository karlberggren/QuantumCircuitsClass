# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 00:05:56 2021

@author: Andres Lombo
"""

import numpy as np
import scipy.integrate as integrate
from scipy.integrate import solve_ivp

from bokeh.driving import count
from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.layouts import layout, column, row
from bokeh.models import Circle, ColumnDataSource, Plot, Button, Slider, CheckboxButtonGroup

ħ = 1  # h = 6.63e-34 J s or 6.58e-16 eV s
       # ħ = h / 2π = 1.05 e -34
π = np.pi
Φₒ = 1  # Φₒ = 2.07 e -15 in reality.
ⅉ = 1j

L = 1
C = 1
LC_V = lambda t, i: 1/2 * L * i**2
LC_dVdx = lambda t, i:  L * i
xlim = 1.5
inital_conditions = [0.25, 0.5, 0.75, 1]
colours_list = ['yellow','blue','red','green']
Δt = 50  # in ms
t_o = 0
paused = True

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

ccs = []
for starting_pos in inital_conditions:
    #                                x_o,    p_o, m_eff, V, dVdx
    ccs.append(Classical_circuit(starting_pos, 0, 1, LC_V, LC_dVdx))
    
data = {
    'x': [cc.x_o for cc in ccs],
    'y': [cc.V(t_o,cc.x_o) for cc in ccs],
    'colours': colours_list,
    'vis': [1.0, 1.0, 1.0, 1.0]
}
points_source = ColumnDataSource(data)

def update_line(xleft,xright,L):
    x = np.linspace(xleft,xright)
    y = 0.5*L*np.square(x)
    return {'x': x, 'y': y}

data = update_line(-xlim,xlim,L) # update initial values for potential
potential_source = ColumnDataSource(data)

# Plot elements
p = figure(title='Simulation of an LC potential', x_range=[-xlim,xlim])
points = p.circle('x', 'y', fill_color='colours', fill_alpha='vis', size=10, line_color=None, source=points_source)
potential = p.line('x', 'y', source=potential_source)

# X-axis slider
slider_xlim = Slider(start=0.1, end=5, step=0.1, value=xlim, title='x-axis limits')
def update_xlim(attr,old,new):
    xlim = round(slider_xlim.value,1)
    p.x_range.start, p.x_range.end = -xlim, xlim
    # Update potential
    potential_source.data = update_line(-xlim,xlim,round(slider_ind.value,1))
slider_xlim.on_change('value',update_xlim)

# Capacitance slider
slider_cap = Slider(start=0.1, end=5, step=0.1, value=C, title='Capacitance')
def update_cap(attr,old,new):
    if paused:
        for cc in ccs:
            cc.m_eff = slider_cap.value
slider_cap.on_change('value',update_cap)

# Inductance slider
slider_ind = Slider(start=0.1, end=5, step=0.1, value=L, title='Inductance')
def update_ind(attr,old,new):
    potential_source.data = update_line(-xlim,xlim,round(slider_ind.value,1))
    if paused:
        for cc in ccs:
            cc.dVdx = lambda t, i:  L * i
            cc.V = lambda t, i: 1/2 * L * i**2
        x = points_source.data['x']
        y = [LC_V(0,i) for i in x]
        points_source.data = {'x': x, 'y': y, 'colours': colours_list, 'vis': points_source.data['vis']}
slider_ind.on_change('value',update_ind)

# Play/Pause button
pause_button = Button(label='► Play')
def animate(event):
    global process_id
    global paused
    if pause_button.label == '► Play':
        paused = False
        pause_button.label = '❚❚ Pause'
        print('Unpaused callback')
        process_id = curdoc().add_periodic_callback(anim_func, 50)
    else:
        paused = True
        pause_button.label = '► Play'
        print('Paused callback')
        curdoc().remove_periodic_callback(process_id)
pause_button.on_click(animate)

# Checkboxes for each point
checkboxes = CheckboxButtonGroup(labels=['1st','2nd','3rd','4th'],active=[0,1,2,3])
def update_checkboxes(attr,old,new):
    vis = [0.0,0.0,0.0,0.0]
    for i in checkboxes.active:
        vis[i] = 1.0
    points_source.data = {
        'x': points_source.data['x'],
        'y': points_source.data['y'],
        'colours': colours_list,
        'vis': vis
        }
checkboxes.on_change('active',update_checkboxes)

# Animation function
def anim_func():
    global t_o
    # First get intial values
    L = slider_ind.value
    C = slider_cap.value
    # Update parameters in model
    for cc in ccs:
        cc.m_eff = C
        cc.dVdx = lambda t, i:  L * i
        cc.V = lambda t, i: 1/2 * L * i**2
    t_f = t_o + Δt / 1000
    x, y = [], []
    for n, cc in enumerate(ccs):
        _, xs, ps = cc.sim((t_o, t_f))
        cc.x_o, cc.p_o = xs[-1], ps[-1]
        x.append(cc.x_o)
        y.append(cc.V(t_f, cc.x_o))
    # Update ColumnDataSource
    points_source.data = {'x': x, 'y': y, 'colours': colours_list, 'vis': points_source.data['vis']}
    t_o += Δt/1000

layout = column(p,slider_xlim,slider_cap,slider_ind,row(pause_button,checkboxes))
curdoc().add_root(layout)
curdoc().title = 'LC-anim'


# from threading import Thread

# from flask import Flask, render_template
# from tornado.ioloop import IOLoop

# from bokeh.embed import server_document
# from bokeh.layouts import column
# from bokeh.models import ColumnDataSource, Slider
# from bokeh.plotting import figure
# from bokeh.sampledata.sea_surface_temperature import sea_surface_temperature
# from bokeh.server.server import Server
# from bokeh.themes import Theme

# app = Flask(__name__)


# def bkapp(doc):
#     df = sea_surface_temperature.copy()
#     source = ColumnDataSource(data=df)

#     plot = figure(x_axis_type='datetime', y_range=(0, 25), y_axis_label='Temperature (Celsius)',
#                   title="Sea Surface Temperature at 43.18, -70.43")
#     plot.line('time', 'temperature', source=source)

#     def callback(attr, old, new):
#         if new == 0:
#             data = df
#         else:
#             data = df.rolling(f"{new}D").mean()
#         source.data = ColumnDataSource.from_df(data)

#     slider = Slider(start=0, end=30, value=0, step=1, title="Smoothing by N Days")
#     slider.on_change('value', callback)

#     doc.add_root(column(slider, plot))

#     doc.theme = Theme(filename="theme.yaml")


# @app.route('/', methods=['GET'])
# def bkapp_page():
#     script = server_document('http://localhost:5006/bkapp')
#     return render_template("embed.html", script=script, template="Flask")


# def bk_worker():
#     # Can't pass num_procs > 1 in this configuration. If you need to run multiple
#     # processes, see e.g. flask_gunicorn_embed.py
#     server = Server({'/bkapp': bkapp}, io_loop=IOLoop(), allow_websocket_origin=["localhost:8000"])
#     server.start()
#     server.io_loop.start()

# Thread(target=bk_worker).start()

# if __name__ == '__main__':
#     print('Opening single process Flask app with embedded Bokeh application on http://localhost:8000/')
#     print()
#     print('Multiple connections may block the Bokeh app in this configuration!')
#     print('See "flask_gunicorn_embed.py" for one way to run multi-process')
#     app.run(port=8000)