# Website source code
If you are reading this, congradulations! Your income potential just rose by 20k because now you are a "wEb DeVelOpEr"

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


## Deploying
### Locally
To deploy the site locally, navigate to `main` and run `app.py` by typing `python app.py` into the command line.
### Google App Engine
To deploy to google app engine, you need to be signed into a Google Cloud account with administrative permissions to a project. You need to download the Gcloud command line tool and authenticate your credentials. Once that is all done, navigate to the `Web_app` directory and run `gcloud app deploy main/main.yaml bokeh1/bokeh1.yaml dispatch.yaml` to deploy all the services. Alternatively, run `gcloud app deploy main/main.yaml  dispatch.yaml` to only deploy the main service. 