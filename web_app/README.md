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
#### Classical LC circuit 
*Need more information about this sandbox*
Implemented using a bokeh server which utilizes callbacks to update the visualization. Because the number of varying parameters is large (k=5), precomputing all possible data points will require an unreasonable number of calculations (n^k). Therefore, callback was prefferd for this task. 
#### Complex wavefunction visualization 
*Need more information about this sandbox*
#### Gaussian Wave-packet
**Learning goal:** introduce the gaussian wavepacket as a wavefunction of special interest and show how the wavefunction and the probaility amplitude change when varying the wavefunction's parameters. 
**Implementation:** plotly with precomputed data. All data points are precomputed and attached to the plot object. A slider controls which traces are visible at any time. 


#### Quantum measurement

#### Diffusion sandbox
*Need more information about this sandbox*