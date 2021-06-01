# Website source code
If you are reading this, congratulations! Your income potential just rose by 20k because now you are a "wEb DeVelOpEr"

## Code Structure 
### General 
The code for the main website is contained in the `main` directory. The other directories in the root directory contain the code for various bokeh sessions which need to be hosted on a public facing server in order to get integrated by the main site. The structure of the current code is:

```
web_app 
    |__dispatch.yaml
    |__ bokeh1
            |__ bokeh1.yaml
            |__ Dockerfile
            |__ LC_circuit_bokeh.py
            |__ requirements.txt
    
    |__ bokeh2 
            |__ bokeh2.yaml
            |__ Dockerfile
            |__ wavevector_measure_bokeh.py
            |__ utils 
                |__ wavevector.py 
                |__ q_operator.py
    |__ main
            |__ main.yaml
            |__ app.py
            |__ requirements.txt
            |__ static
                    |__ figures
                    |__ css
                            |__ main.css
            |__ templates
                    |__ All the HTML files
            |__ utils
                    |__ Sandbox files

```
The `dispatch.yaml` file is responsible for defining service allocation for the Google app deployment. To be perfectly honest, I don't have the slightest idea how it works or what it does, but I know through trial and error (lots and lots of error) that it makes things work. I'll explain how to use it in the [Deploying on Google App Engine](https://github.com/karlberggren/QuantumCircuitsClass/tree/main/web_app#deploying) section. 

### Main
#### Python backend 
The script in `main/app.py` uses `flask` to create an API which serves the site. Each page on the site is defined by a decorated function of the following structure:
```
@app.route('/foo', methods=['GET', 'POST'])
def foobar():
    # do stuff ... or not ...
    return render_template("foo.html", html_args = args)
```
Here, the 0th aegument in the `@app.route` decorator is the url extension that the specific page will have. The function can do stuff to generate the html args, and at the end, it renders an html template for that page. The html_args are optional. If present, they are passed to place holders in the `"foo.html"` and integrated into the page seamlessly. The args can be anything from a string to a javascript object. 

#### Templates 
Each page on the website has an html page. The naming scheme of the HTML pages should match up to the URL extension. Otherwise, this code base will be a complete mess once all chapters are in place.
Each HTML page includes a base html which contains all the necessary packages and a navbar html which does the navigation bar at the top of the page. Right now, the navigation bar isn't working. It should allow the reader to jump to different topics in the page instead of scrolling. 
HTML pages can have placeholders. Use `{{html_arg1}}` to define the place holder for argument 1 in the HTML. You fill the place holder with the correct argument by doing  `return render_template("foo.html", html_arg1 = arg1)` in the `app.py` file. **Notice: the name to the left of the equal sign in the render_template must match up to the name inside the double curly braces in the html file.** I literally spent days trying to figure out why my code wasn't working, just to realize that I mispelled the name in the `render_template`. The debugger does not find this error for you.  
The double curly braces can be used also to add images or hyperlinks. 
For example `<img src="{{url_for('static', filename='MIT_c.jpg')}}" alt="MIT" width="270" height="100" align="left">` adds an image named `"MIT_c.jpg"` to the page. `<a href="{{ url_for('classical_LC') }}">Classical LC Circuits</a></li>` is used to add a hyperlink to the Classical LC page. Notice: the argument inside `url_for()` is the URL extension of that page. 

#### Static
All the figures included in the text must be placed in the static folder. 
I really don't know how CSS works. So, whenever I have a question, I just google it and copy and paste the code.

### Bokeh 
#### Bokeh1
Bokeh1 service contains the interactive LC circuit simulation. The main entry point for the simulation is in the `LC_circuit_bokeh.py` script. The Google app engine uses `bokeh1.yaml` and `Dockerfile` to define the production environment.

#### Bokeh2 
Bokeh2 service contains the interactive wavefunction measurement simulation. The main entry point for the simulation is in the `wavevector_measure_bokeh.py` script. The Google app engine uses `bokeh2.yaml` and `Dockerfile` to define the production environment.

## Deploying
### Locally
To deploy the bokeh simulation, navigate to the bokeh directory and run `bokeh serve --<port number> <script>` or `python -m bokeh serve --<port number> <script>` if the former doesn't work. if you don't specify the port number, it will get deployed to port 5006.

To deploy the main site locally, navigate to `main` and uncomment line 68 and comment line 67 in `app.py`. In `web_app/main` and run `app.py` by typing `python app.py` into the command line.
### Google App Engine
#### If you already have a Google Cloud Project set up and the SDK downloaded
Navigate to the `Web_app` directory and run `gcloud app deploy main/main.yaml bokeh1/bokeh1.yaml bokeh2/bokeh2.yaml dispatch.yaml` to deploy all the services. Alternatively, run `gcloud app deploy main/main.yaml  dispatch.yaml` to only deploy the main service. 

#### Getting started with App Engine
1. Navigate to your [cloud console](https://console.cloud.google.com).
2. Create a new project by clicking on the 'select project' dropdown menu on the banner and choose to make a new project.
3. Enable billing for the app engine in the newly created project. 
4. Follow [these](https://cloud.google.com/sdk/docs/install) instructions to download and install the Gcloud SDK. 
5. Once installed, you'll need to authenticate your credentials and select a project to link the SDK. 
6. Now you can navigate to the `Web_app` directory and run `gcloud app deploy main/main.yaml bokeh1/bokeh1.yaml bokeh2/bokeh2.yaml dispatch.yaml` to deploy all the services. Alternatively, run `gcloud app deploy main/main.yaml  dispatch.yaml` to only deploy the main service. 

## Sandboxes 
### Classical LC circuit 
*Need more information about this sandbox*
Implemented using a bokeh server which utilizes callbacks to update the visualization. Because the number of varying parameters is large (k=5), precomputing all possible data points will require an unreasonable number of calculations (n^k). Therefore, callback was prefferd for this task. 
### Complex wavefunction visualization 
*Need more information about this sandbox*
### Gaussian Wave-packet
**Learning goal:** introduce the gaussian wavepacket as a wavefunction of special interest and show how the wavefunction and the probaility amplitude change when varying the wavefunction's parameters.
**Implementation:** plotly with pre-computed data. All data points are pre-computed and attached to the plot object. A slider controls which traces are visible at any time. The changeable variables: wavenumber (momentum), mean and spread.   
The below code snippet creates a gaussian wavepacket sandbox where the wavenumber can be varied using a slider. The full code can be found [here](https://github.com/karlberggren/QuantumCircuitsClass/blob/main/web_app/main/utils/plotly_slider.py).
```
import plotly.graph_objects as go
import numpy as np

# Create figure
fig_k = go.Figure()

# Initialize x range and parameters
x=np.arange(-5, 5.05, 0.05)
k= np.linspace(-5, 5, 21)
phi_0 = 0
sig = 1

# Iterate over all k values. For each k value, compute the complex wavefunction, real part, imaginary part, and probability amplitude.  
for k_i in np.arange(-5, 5, 0.5):
    psi = np.multiply(np.exp(1j*k_i*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))))
    # add wavefunction trace to figure
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=x,
            y=np.imag(psi), 
            z=np.real(psi),
            mode="lines"))
    # add real projection of wavefunction trace to figure
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=x,
            y=-np.ones(psi.shape),
            z=np.real(psi),  
            mode="lines"))
    # add imaginary projection of wavefunction trace to figure
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= x,
            y= np.imag(psi), 
            z= -np.ones(psi.shape), 
            mode="lines"))

x2 = np.arange(-5, 5.05, 0.05)
y2 =-np.ones(x2.shape)
z2 = np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi)))
# add probability amplitude trace to figure
fig_k.add_trace(go.Scatter3d(visible=True, x=x2, y=y2, z=z2, name="prob amplitude", mode="lines", line=dict(color='black', width=2)))

# Make the traces corresponding to k=0 visible by default
fig_k.data[30].visible = True
fig_k.data[31].visible = True
fig_k.data[32].visible = True

# Create and add slider
steps = []
for i in range(0, len(fig_k.data)-1, 3):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig_k.data)},
              {"title": "k= " + str(k[i//3])}],  # layout attribute
        label=str(k[i//3])
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i+1'th trace to "visible"
    step["args"][0]["visible"][i+2] = True  # Toggle i+2'th trace to "visible"
    step["args"][0]["visible"][-1] = True   # Toggle the prob amplitude trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Wave number: "},
    pad={"t": 50},
    steps=steps
)]

# add slider to image object
fig_k.update_layout(
    sliders=sliders
)

# Update latout 
fig_k.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=4, range=[-1,1],),
        zaxis = dict(nticks=4, range=[-1,1],),
        xaxis_title='Phi',
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
# Update latout 
fig_k.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=2, y=1, z=1))

fig_k.show()
fig_k.write_html("wavefunction_changing_k.html")
```

### Quantum measurement

#### Diffusion sandbox
*Need more information about this sandbox*