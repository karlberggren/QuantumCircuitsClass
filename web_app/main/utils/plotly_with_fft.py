import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.fftpack import fft, ifft, fftfreq


# Changing K plot
# Create figure
fig_k = make_subplots(
    rows=1, cols=2,
    shared_xaxes=False,
    horizontal_spacing=0.02,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])


# Add traces, one for each slider step
k= np.linspace(-10, 10, 21)
phi_0 = 0
sig = 0.5
x=np.arange(-20, 20.01, 0.01)

for k_i in np.arange(-10, 11, 1):
    psi = np.multiply(np.exp(1j*k_i*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))))
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=x,
            y=np.imag(psi), 
            z=np.real(psi),
            mode="lines"), row=1, col=1)
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=x,
            y=-2*np.ones(psi.shape),
            z=np.real(psi),  
            mode="lines"), row=1, col=1)
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= x,
            y= np.imag(psi), 
            z= -2*np.ones(psi.shape), 
            mode="lines"), row=1, col=1)
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=1),
            name="imaginary part",
            x= x,
            y= -2*np.ones(psi.shape), 
            z= np.abs(psi), 
            mode="lines"), row=1, col=1)

for k_i in np.arange(-10, 11, 1):
    psi_fft = fft(np.multiply(np.exp(1j*k_i*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi)))))
    xf = fftfreq(len(x), x[1] - x[0])
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y=np.imag(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            z=np.real(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))),
            mode="lines"), row=1, col=2)
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            #x=xf,
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y=-10*np.ones(psi_fft.shape),
            #z=np.real(psi_fft),
            z=np.real(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))),  
            mode="lines"), row=1, col=2)
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            #x= xf,
            #y= np.imag(psi_fft), 
            y=np.imag(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))),
            z= -200*np.ones(psi_fft.shape), 
            mode="lines"), row=1, col=2)
    fig_k.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=1),
            name="imaginary part",
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            #x= xf,
            y= -10*np.ones(psi_fft.shape), 
            z= np.abs(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            mode="lines"), row=1, col=2)

fig_k.data[40].visible = True
fig_k.data[41].visible = True
fig_k.data[42].visible = True
fig_k.data[43].visible = True

fig_k.data[(len(fig_k.data))//2+40].visible = True
fig_k.data[(len(fig_k.data))//2+41].visible = True
fig_k.data[(len(fig_k.data))//2+42].visible = True
fig_k.data[(len(fig_k.data))//2+43].visible = True

# Create and add slider
steps = []
for i in range(0, (len(fig_k.data))//2, 4):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig_k.data)}],  # layout attribute
        label=str(k[i//4])
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+3] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][(len(fig_k.data))//2+i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][(len(fig_k.data))//2+i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][(len(fig_k.data))//2+i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][(len(fig_k.data))//2+i+3] = True  # Toggle i'th trace to "visible"
    #step["args"][0]["visible"][-1] = True 
    #step["args"][0]["visible"][-2] = True 
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "Wave number: "},
    pad={"t": 33},
    steps=steps
)]

fig_k.update_layout(
    sliders=sliders
)

fig_k.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2,2],),
        xaxis_title='Phi',
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
    #width=1000,
    #margin=dict(r=20, l=10, b=10, t=10))
fig_k.update_layout(
    scene2 = dict(
        xaxis = dict(nticks=10, range=[-3, 3]),
        yaxis = dict(nticks=4, range=[-10,10]),
        zaxis = dict(nticks=4, range=[-200, 200]),
        xaxis_title='Q',
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
fig_k.update_layout(scene_aspectmode='manual',
                  scene_aspectratio=dict(x=2, y=1, z=1), 
                  scene2_aspectmode='manual',
                  scene2_aspectratio=dict(x=2, y=1, z=1), 
                  showlegend=False,
                  margin=dict(l=10, r=10, t=5, b=5), 
                  scene_camera=dict(eye=dict(x=1, y=2.4, z=0.4)),
                  scene2_camera=dict(eye=dict(x=1, y=2.4, z=0.4)))

fig_k.show(config = {'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False})
fig_k.write_html("wavefunction_changing_k_with_fft.html", config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False})

# Changing sig plot
# Create figure
fig_sig = make_subplots(
    rows=1, cols=2,
    shared_xaxes=True,
    horizontal_spacing=0.02,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

# Add traces, one for each slider step
k = 3
sig_arr = np.logspace(-0.8, 0.5, 4)
phi_0 = 0
sig = 1
x=np.arange(-20, 20.01, 0.01)

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
            mode="lines"), row=1, col=1)
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=x,
            y=-2*np.ones(psi.shape),
            z=np.real(psi),  
            mode="lines"), row=1, col=1)
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= x,
            y= np.imag(psi), 
            z= -2*np.ones(psi.shape), 
            mode="lines"), row=1, col=1)
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False, 
            x=x, 
            y= -2*np.ones(psi.shape), 
            z=np.abs(psi), 
            name="prob amplitude", mode="lines", line=dict(color='black', width=2)), row=1, col=1)

for sig_i in sig_arr:
    psi_fft = fft(np.multiply(np.exp(1j*k*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig_i**2))/(sig_i*np.sqrt(2*np.pi)))))
    xf = fftfreq(len(x), x[1] - x[0])
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y=np.imag(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))),
            z=np.real(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            mode="lines"), row=1, col=2)
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y=-9*np.ones(psi_fft.shape),
            z=np.real(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))),  
            mode="lines"), row=1, col=2)
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y= np.imag(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            z= -320*np.ones(psi_fft.shape), 
            mode="lines"), row=1, col=2)
    fig_sig.add_trace(
        go.Scatter3d(
            visible=False, 
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])), 
            y= -9*np.ones(psi_fft.shape), 
            z=np.abs(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            name="prob amplitude", mode="lines", line=dict(color='black', width=2)), row=1, col=2)


# Make 10th trace visible
fig_sig.data[4].visible = True
fig_sig.data[5].visible = True
fig_sig.data[6].visible = True
fig_sig.data[7].visible = True
fig_sig.data[len(fig_sig.data)//2+4].visible = True
fig_sig.data[len(fig_sig.data)//2+5].visible = True
fig_sig.data[len(fig_sig.data)//2+6].visible = True
fig_sig.data[len(fig_sig.data)//2+7].visible = True

# Create and add slider
steps = []
for i in range(0, len(fig_sig.data)//2, 4):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig_sig.data)}],  # layout attribute
        label=str(round(sig_arr[i//4],2))
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+3] = True 
    step["args"][0]["visible"][len(fig_sig.data)//2+i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][len(fig_sig.data)//2+i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][len(fig_sig.data)//2+i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][len(fig_sig.data)//2+i+3] = True 
    steps.append(step)

sliders = [dict(
    active=1,
    currentvalue={"prefix": "Standard diviation: "},
    pad={"t": 33},
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
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
    #margin=dict(r=20, l=10, b=10, t=10))
fig_sig.update_layout(
    scene2 = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=4, range=[-9, 9],),
        zaxis = dict(nticks=4, range=[-320, 320],),
        xaxis_title='Q',
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
fig_sig.update_layout(scene_aspectmode='manual',
    scene_aspectratio=dict(x=2, y=1, z=1), 
                  scene2_aspectmode='manual',
                  scene2_aspectratio=dict(x=2, y=1, z=1), 
                  showlegend=False, 
                  margin=dict(l=10, r=10, t=5, b=5),
                  scene_camera=dict(eye=dict(x=1, y=2.4, z=0.4)), 
                  scene2_camera=dict(eye=dict(x=1, y=2.4, z=0.4))
                  #width=1000
                  )


fig_sig.show(config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False})
fig_sig.write_html("wavefunction_changing_sig_with_fft.html", config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False})

# Changing sig plot
# Create figure
fig_mean = make_subplots(
    rows=1, cols=2,
    shared_xaxes=True,
    horizontal_spacing=0.02,
    specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]])

# Add traces, one for each slider step
k = 3
sig = 1
phi_0 = np.arange(-3, 3.5, 0.5)
sig = 1
x=np.arange(-20, 20.01, 0.01)


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
            mode="lines"), row=1, col=1)
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=x,
            y=-np.ones(psi.shape), 
            z=np.real(psi), 
            mode="lines"), row=1, col=1)
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= x,
            y= np.imag(psi), 
            z= -np.ones(psi.shape), 
            mode="lines"), row=1, col=1)
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False, 
            x=x, 
            y= -np.ones(psi.shape), 
            z=np.abs(psi), 
            name="prob amplitude", mode="lines", line=dict(color='black', width=2)), row=1, col=1)

for phi_0_i in phi_0:
    psi_fft = fft(np.multiply(np.exp(1j*k*x), np.sqrt(np.exp(-np.power(x-phi_0_i, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi)))))
    xf = fftfreq(len(x), x[1] - x[0])
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="black", width=2),
            name="wavefunction",
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y=np.imag(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            z=np.real(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            mode="lines"), row=1, col=2)
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="red", width=1),
            name="real part",
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y=-290*np.ones(psi_fft.shape), 
            z=np.real(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            mode="lines"), row=1, col=2)
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False,
            line=dict(color="orange", width=1),
            name="imaginary part",
            x= np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])),
            y= np.imag(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            z= -410*np.ones(psi_fft.shape), 
            mode="lines"), row=1, col=2)
    fig_mean.add_trace(
        go.Scatter3d(
            visible=False, 
            x=np.concatenate((xf[len(xf)//2+1:], [xf[0]], xf[1:len(xf)//2])), 
            y= -290*np.ones(psi_fft.shape), 
            z=np.abs(np.concatenate((psi_fft[len(xf)//2+1:],[psi_fft[0]], psi_fft[1:len(xf)//2]))), 
            name="prob amplitude", mode="lines", line=dict(color='black', width=2)), row=1, col=2)



# Make 10th trace visible
fig_mean.data[24].visible = True
fig_mean.data[25].visible = True
fig_mean.data[26].visible = True
fig_mean.data[27].visible = True
fig_mean.data[len(fig_mean.data)//2+24].visible = True
fig_mean.data[len(fig_mean.data)//2+25].visible = True
fig_mean.data[len(fig_mean.data)//2+26].visible = True
fig_mean.data[len(fig_mean.data)//2+27].visible = True

# Create and add slider
steps = []
for i in range(0, len(fig_mean.data)//2, 4):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig_mean.data)}],  # layout attribute
        label=str(phi_0[i//4])
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][i+3] = True 
    step["args"][0]["visible"][len(fig_mean.data)//2+i] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][len(fig_mean.data)//2+i+1] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][len(fig_mean.data)//2+i+2] = True  # Toggle i'th trace to "visible"
    step["args"][0]["visible"][len(fig_mean.data)//2+i+3] = True 
    steps.append(step)

sliders = [dict(
    active=6,
    currentvalue={"prefix": "Mean: "},
    pad={"t": 33},
    steps=steps
)]

fig_mean.update_layout(
    sliders=sliders
)

fig_mean.update_layout(
    scene = dict(
        xaxis = dict(nticks=10, range=[-5,5],),
        yaxis = dict(nticks=4, range=[-2,2],),
        zaxis = dict(nticks=4, range=[-2,2],),
               xaxis_title='Phi',
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
    #margin=dict(r=20, l=10, b=10, t=10))
fig_mean.update_layout(
    scene2 = dict(
        xaxis = dict(nticks=10, range=[-3,3],),
        yaxis = dict(nticks=4, range=[-290, 290],),
        zaxis = dict(nticks=4, range=[-430, 430],),
        xaxis_title='Q',
        yaxis_title='Imaginary', 
        zaxis_title='Real',))
fig_mean.update_layout(scene_aspectmode='manual',
    scene_aspectratio=dict(x=2, y=1, z=1), 
                  scene2_aspectmode='manual',
                  scene2_aspectratio=dict(x=2, y=1, z=1), 
                  showlegend=False, 
                  margin=dict(l=10, r=10, t=5, b=5), 
                  scene_camera=dict(eye=dict(x=1, y=2.4, z=0.4)),
                  scene2_camera=dict(eye=dict(x=1, y=2.4, z=0.4))
                  #width=1000
                  )

fig_mean.show(config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False})
fig_mean.write_html("wavefunction_changing_mu_with_fft.html", config={'displayModeBar': False, 'displaylogo': False, 'scrollZoom': False})