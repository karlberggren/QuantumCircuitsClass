import numpy as np
#import pandas as pd

from bokeh.embed import components, server_document
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, HoverTool, PrintfTickFormatter, Slider, TextInput
from bokeh.plotting import figure
from bokeh.transform import factor_cmap
from bokeh.io import curdoc
from bokeh.client import pull_session
from bokeh.embed import server_session
from bokeh.embed import server_document

from threading import Thread
from tornado.ioloop import IOLoop
from bokeh.server.server import Server

from flask import Flask, render_template, request

palette = ['#ba32a0', '#f85479', '#f8c260', '#00c2ba']

chart_font = 'Helvetica'
chart_title_font_size = '16pt'
chart_title_alignment = 'center'
axis_label_size = '14pt'
axis_ticks_size = '12pt'
default_padding = 30
chart_inner_left_padding = 0.015
chart_font_style_title = 'bold italic'


def palette_generator(length, palette):
    int_div = length // len(palette)
    remainder = length % len(palette)
    return (palette * int_div) + palette[:remainder]


def plot_styler(p):
    p.title.text_font_size = chart_title_font_size
    p.title.text_font  = chart_font
    p.title.align = chart_title_alignment
    p.title.text_font_style = chart_font_style_title
    p.y_range.start = 0
    p.x_range.range_padding = chart_inner_left_padding
    p.xaxis.axis_label_text_font = chart_font
    p.xaxis.major_label_text_font = chart_font
    p.xaxis.axis_label_standoff = default_padding
    p.xaxis.axis_label_text_font_size = axis_label_size
    p.xaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_text_font = chart_font
    p.yaxis.major_label_text_font = chart_font
    p.yaxis.axis_label_text_font_size = axis_label_size
    p.yaxis.major_label_text_font_size = axis_ticks_size
    p.yaxis.axis_label_standoff = default_padding
    p.toolbar.logo = None
    p.toolbar_location = None


app = Flask(__name__)

@app.route('/',methods=['GET'] )
def home():
    return render_template("home.html")

@app.route('/Classical_LC',methods=['GET'] )
def classical_LC():
    script = server_document('https://bokeh1-dot-test-deploying-320720.ue.r.appspot.com/LC_circuit_bokeh')
    #script = server_document('https://bokeh1-dot-quantum-circuits-306715.ue.r.appspot.com/LC_circuit_bokeh')
    #script = server_document('http://localhost:5006/LC_circuit_bokeh')
    return render_template("classical_LC.html", script=script)

@app.route('/teaching_staff', methods=['GET'])
def teaching_staff():
    return render_template("teaching_staff.html")

@app.route('/schrodinger',methods=['GET'] )
def schrodinger():
    return render_template("schrodinger.html")

@app.route('/probability',methods=['GET'] )
def probability():
    Probability_dens, Quantum_state = redraw(0)
    script_Probability_dens, div_Probability_dens = components(Probability_dens)
    script_Quantum_state , div_Quantum_state  = components(Quantum_state )
    return render_template("probability.html",     
            div_Probability_dens=div_Probability_dens,
            script_Probability_dens=script_Probability_dens,
            div_Quantum_state=div_Quantum_state,
            script_Quantum_state=script_Quantum_state)

@app.route('/Prob_amplitiude', methods=['GET', 'POST'])
def Prob_amplitiude():
    selected_class = request.form.get('dropdown-select')
    #script = server_document('https://bokeh2-dot-quantum-circuits-306715.ue.r.appspot.com/wavevector_measure_bokeh')
    script = server_document('http://localhost:5006/wavevector_measure_bokeh')
    return render_template('index.html', wavevector_measurement =script, selected_class=selected_class)


def animated_interactive(data, pass_class):
    # with pull_session(url="http://localhost:5006/LC_circuit_bokeh") as session:

    #     # update or customize that session
    #     session.document.roots[0].children[1].title.text = "Special sliders for a specific user!"

    #     # generate a script to load the customized session
    #     script = server_session(session_id=session.id, url='http://localhost:5006/LC_circuit_bokeh')
    #     return script
    return "place holder"

def Probability_dens(mu, sig, pass_class, cpalette=palette):
    x = np.linspace(-5, 5, 50)
    y = 1/(sig*np.sqrt(2*np.pi))*np.exp(-0.5*np.power((x-mu)/sig, 2))
    
    source = ColumnDataSource(data={
        'x': x,
        'y':y
    })

    hover_tool = HoverTool(
        tooltips=[('x', '@x'), ('y', '@y')]
    )
    
    
        
    p = figure(tools=[hover_tool], plot_height=200, title='Guassian')
    p.line(x='x', y='y', source=source, width=0.9)
    
    plot_styler(p)
    p.xaxis.ticker = source.data['x']
    p.sizing_mode = 'scale_width'
    
    return p

    
def Quantum_state(dataset, pass_class, color=palette[1]):
    x = np.random.rand(25)
    y = np.random.rand(25)
    
    source = ColumnDataSource({
        'x': x,
        'y': y
    })

    p = figure(plot_height=200, title='Age Histogram')
    p.circle('x', y='y', source=source,
            fill_color=color)

    plot_styler(p)
    p.sizing_mode = 'scale_width'

    return p


def redraw(p_class):
    p_dens = Probability_dens(0, 1, p_class)
    q_state = Quantum_state(0, p_class)
    return (
        p_dens,
        q_state
    )

def bk_worker():
    # Can't pass num_procs > 1 in this configuration. If you need to run multiple
    # processes, see e.g. flask_gunicorn_embed.py
    server = Server({'/bkapp': bkapp}, io_loop=IOLoop(), allow_websocket_origin=["localhost:8000"])
    server.start()
    server.io_loop.start()


if __name__ == '__main__':
    app.run(debug=True, port=8080)