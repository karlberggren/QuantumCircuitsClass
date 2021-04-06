# Website source code
If you are reading this, congradulations! Your income potential just rose by 20k because now you are a "web developer"

## Structure 
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

### Main
#### Python backend 
The script in `main/app.py` uses `flask` to create an API which serves the site. Each page on the site is defined by a decorated function of the following structure:
```
@app.route('/foo', methods=['GET', 'POST'])
def foobar():
    # do stuff ... or not ...
    return render_template("foo.html", html_args = args)
```
Here, the 0th aegument in the `@app.route' decorator is the url extension that the specific page will have. The function can do stuff to generate the html args, and at the end, it renders an html template for that page. The html_args are optional. If present, they are passed to place holders in the "foo.html" and integrated into the page seamlessly. The args can be anything from a string to a javascript object. 


## Deploying
To deploy tShe site locally, navigate to `main` and run `app.py` by typing `python app.py` into the command line.

