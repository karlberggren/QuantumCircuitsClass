import numpy as np
#import pandas as pd
from bokeh.embed import server_document
from bokeh.embed import server_document
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/',methods=['GET'] )
def home():
    return render_template("home.html")

@app.route('/teaching_staff', methods=['GET'])
def teaching_staff():
    return render_template("teaching_staff.html")

@app.route('/classical_LC',methods=['GET'] )
def classical_LC():
    selected_class = request.form.get('dropdown-select')
    #script = server_document('https://bokeh1-dot-quantum-explorations.uk.r.appspot.com/LC_circuit_bokeh')
    #script = server_document('http://localhost:5006/LC_circuit_bokeh')
    return render_template("classical_LC.html")

@app.route('/probability',methods=['GET'] )
def probability():
    selected_class = request.form.get('dropdown-select')
    #Probability_dens, Quantum_state = redraw(0)
    #script_Probability_dens, div_Probability_dens = components(Probability_dens)
    #script_Quantum_state , div_Quantum_state  = components(Quantum_state )
    return render_template("probability.html", selected_class=selected_class)

@app.route('/prob_amplitude', methods=['GET', 'POST'])
def prob_amplitude():
    selected_class = request.form.get('dropdown-select')
    return render_template('prob_amplitude.html', selected_class=selected_class)

@app.route('/measurement', methods=['GET', 'POST'])
def measurement():
    selected_class = request.form.get('dropdown-select')
    script = server_document('https://bokeh-dot-quantum-explorations.uk.r.appspot.com/wavevector_measure_bokeh')
    #script = server_document('http://localhost:5006/wavevector_measure_bokeh')
    return render_template('measurement.html', wavevector_measurement =script, selected_class=selected_class)

@app.route('/schrodinger',methods=['GET'] )
def schrodinger():
    selected_class = request.form.get('dropdown-select')
    script1 = server_document('https://bokeh-dot-quantum-explorations.uk.r.appspot.com/diffusion1')
    script2 = server_document('https://bokeh-dot-quantum-explorations.uk.r.appspot.com/diffusion2')
    #script1 = server_document('http://localhost:5006/diffusion1')
    #script2 = server_document('http://localhost:5006/diffusion2')
    return render_template("schrodinger.html",script1=script1,script2=script2)


if __name__ == '__main__':
    app.run(debug=True, port=8080)