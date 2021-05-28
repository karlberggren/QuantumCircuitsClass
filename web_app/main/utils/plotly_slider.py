import plotly.graph_objects as go
import numpy as np


# Changing K plot
# Create figure
fig_k = go.Figure()

# Add traces, one for each slider step
k= np.linspace(-5, 5, 21)
phi_0 = 0
sig = 1
x=np.arange(-5, 5.05, 0.05)

for k_i in np.arange(-5, 5, 0.5):
    psi = np.multiply(np.exp(1j*k_i*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))))
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=x,
            y=np.imag(psi), 
            z=np.real(psi),
            mode="lines"))
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=x,
            y=-np.ones(psi.shape),
            z=np.real(psi),  
            mode="lines"))
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
#fig2 = px.line_3d(df2, x="x", y="y", z="z",color="z")
fig_k.add_trace(go.Scatter3d(visible=True, x=x2, y=y2, z=z2, name="prob amplitude", mode="lines", line=dict(color='black', width=2)))
# Make 10th trace visible
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
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][-1] = True 
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Wave number: "},
    pad={"t": 50},
    steps=steps
)]

fig_k.update_layout(
    sliders=sliders
)

fig_k.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=4, range=[-1,1],),
        zaxis = dict(nticks=4, range=[-1,1],),
        xaxis_title='Phi',
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
    #width=1000,
    #margin=dict(r=20, l=10, b=10, t=10))
fig_k.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=2, y=1, z=1))

fig_k.show()
fig_k.write_html("wavefunction_changing_k.html")

# Changing sig plot
# Create figure
fig_sig = go.Figure()

# Add traces, one for each slider step
k = 3
sig_arr = np.logspace(-0.71, 0.5, 4)
phi_0 = 0
sig = 1
x=np.arange(-5, 5.05, 0.05)

for sig_i in sig_arr:
    psi = np.multiply(np.exp(1j*k*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig_i**2))/(sig_i*np.sqrt(2*np.pi))))
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=x,
            y=np.imag(psi),
            z=np.real(psi), 
            mode="lines"))
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=x,
            y=-2*np.ones(psi.shape),
            z=np.real(psi),  
            mode="lines"))
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= x,
            y= np.imag(psi), 
            z= -2*np.ones(psi.shape), 
            mode="lines"))
    z2 = np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig_i**2))/(sig_i*np.sqrt(2*np.pi)))
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False, 
            x=x, 
            y= -2*np.ones(x2.shape), 
            z=z2, 
            name="prob amplitude", mode="lines", line=dict(color='black', width=2)))


# Make 10th trace visible
fig_sig.data[4].visible = True
fig_sig.data[5].visible = True
fig_sig.data[6].visible = True
fig_sig.data[7].visible = True

# Create and add slider
steps = []
for i in range(0, len(fig_sig.data), 4):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig_sig.data)},
              {"title": "Sigma: " + str(sig_arr[i//4])}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+3] = True 
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Standard diviation: "},
    pad={"t": 40},
    steps=steps
)]

fig_sig.update_layout(
    sliders=sliders
)

fig_sig.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2,2],),
        xaxis_title='Phi',
        yaxis_title='Real',
        zaxis_title='Imaginary'))
    #width=1000,
    #margin=dict(r=20, l=10, b=10, t=10))
fig_sig.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=2, y=1, z=1))

fig_sig.show()
fig_sig.write_html("wavefunction_changing_sig.html")

# Changing sig plot
# Create figure
fig_mean = go.Figure()

# Add traces, one for each slider step
k = 3
sig = 1
phi_0 = np.arange(-3, 3.5, 0.5)
sig = 1
x=np.arange(-5, 5.05, 0.05)

for phi_0_i in phi_0:
    psi = np.multiply(np.exp(1j*k*x), np.sqrt(np.exp(-np.power(x-phi_0_i, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))))
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=x,
            y=np.imag(psi), 
            z=np.real(psi), 
            mode="lines"))
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=x,
            y=-np.ones(psi.shape), 
            z=np.real(psi), 
            mode="lines"))
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= x,
            y= np.imag(psi), 
            z= -np.ones(psi.shape), 
            mode="lines"))
    z2 = np.sqrt(np.exp(-np.power(x-phi_0_i, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi)))
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False, 
            x=x, 
            y= -np.ones(x2.shape), 
            z=z2, 
            name="prob amplitude", mode="lines", line=dict(color='black', width=2)))


# Make 10th trace visible
fig_mean.data[24].visible = True
fig_mean.data[25].visible = True
fig_mean.data[26].visible = True
fig_mean.data[27].visible = True

# Create and add slider
steps = []
for i in range(0, len(fig_mean.data), 4):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig_mean.data)},
              {"title": "Phi_0= " + str(phi_0[i//4])}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+3] = True 
    steps.append(step)

sliders = [dict(
    active=6,
    currentvalue={"prefix": "Standard diviation: "},
    pad={"t": 40},
    steps=steps
)]

fig_mean.update_layout(
    sliders=sliders
)

fig_mean.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=4, range=[-1,1],),
        zaxis = dict(nticks=4, range=[-1,1],),
        xaxis_title='Phi',
        yaxis_title='Real',
        zaxis_title='Imaginary'))
    #width=1000,
    #margin=dict(r=20, l=10, b=10, t=10))
fig_mean.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=2, y=1, z=1))

fig_mean.show()
fig_mean.write_html("wavefunction_changing_mu.html")