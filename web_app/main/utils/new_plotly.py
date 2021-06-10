
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


## changing k figure
k= np.linspace(-5, 5, 21)
phi_0 = 0
sig = 1
x = np.linspace(-5, 5, 400)
data = {'phi': np.array([]), 'real': np.array([]), 'imaginary': np.array([]), 'k': np.array([])}

fig1 = go.Figure()
for i, k_i in enumerate(k):
    k_arr = np.array([k_i]*len(x))
    psi = np.multiply(np.exp(1j*k_i*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))))
    y = np.real(psi)
    z = np.imag(psi)
    fig1.add_trace(go.Scatter3d(visible=False,
    x=psi, y=y, z=z, name="wavefunction", mode="lines", line=dict(color='blue', width=1)))
    #data['phi'] = np.append(data['phi'], x)
    #data['real'] = np.append(data['real'], y)
    #data['imaginary'] = np.append(data['imaginary'], z)
    #data['k'] = np.append(data['k'], k_arr)

#data['trace'] = ["wavefunction" for i in range(len(data['phi']))]


#df = pd.DataFrame(data=data)
#fig = px.line_3d(df, x="phi", y="real", z="imaginary", animation_frame="k", color="trace")
fig1.add_trace(go.Scatter3d(visible=False,
x=data['phi'], y=data['real'], z=data['imaginary'], 
name="wavefunction", mode="lines", line=dict(color='blue', width=1)))
fig1.data[10].visible = True
steps = []
for i in range(len(fig1.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig1.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=10,
    currentvalue={"prefix": "K: "},
    pad={"t": 50},
    steps=steps
)]

fig1.update_layout(
    sliders=sliders
)

#df2 = pd.DataFrame(dict(x=np.linspace(-5, 5, 300), y=np.exp(-np.power(x-phi_0, 2)/(2*sig[2]**2))/(sig[2]*np.sqrt(2*np.pi)), z=np.zeros(x.shape)))
x2 = np.linspace(-5, 5, 300)
y2 = np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))
z2 =-np.ones(x2.shape)
#fig2 = px.line_3d(df2, x="x", y="y", z="z",color="z")
fig1.add_trace(go.Scatter3d(x=x2, y=y2, z=z2, name="pdf", mode="lines", line=dict(color='red', width=1)))


fig1["layout"].pop("updatemenus") # optional, drop animation buttons
fig1.show()
fig1.write_html("wavefunction_changing_k.html")

# ## changing sig
# k = 3
# sig = np.logspace(-1, 1, 5)
# data2 = {'phi': np.array([]), 'real': np.array([]), 'imaginary': np.array([]), 'sig': np.array([])}
# for i, sig_i in enumerate(sig):
#     sig_arr = np.array([sig_i]*len(x))
#     psi = np.multiply(np.exp(1j*k*x), np.sqrt(np.exp(-np.power(x-phi_0, 2)/(2*sig_i**2))/(sig_i*np.sqrt(2*np.pi))))
#     y = np.real(psi)
#     z = np.imag(psi)
#     data2['phi'] = np.append(data2['phi'], x)
#     data2['real'] = np.append(data2['real'], y)
#     data2['imaginary'] = np.append(data2['imaginary'], z)
#     data2['sig'] = np.append(data2['sig'], sig_arr)

# data2['trace'] = ["wavefunction" for i in range(len(data2['phi']))]

# for i, sig_i in enumerate(sig):
#     sig_arr = np.array([sig_i]*len(x))
#     y = np.exp(-np.power(x-phi_0, 2)/(2*sig_i**2))/(sig_i*np.sqrt(2*np.pi))
#     z = -np.ones(x.shape)
#     data2['phi'] = np.append(data2['phi'], x)
#     data2['real'] = np.append(data2['real'], y)
#     data2['imaginary'] = np.append(data2['imaginary'], z)
#     data2['sig'] = np.append(data2['sig'], sig_arr)

# data2['trace'] = data2['trace'] + ["pdf" for i in range(len(data2['phi']) - len(data2['trace']))]
# df2 = pd.DataFrame(data = data2)
# fig2 = px.line_3d(df2, x="phi", y="real", z="imaginary", animation_frame="sig", color="trace")
# fig2["layout"].pop("updatemenus") # optional, drop animation buttons
# fig2.write_html("wavefunction_changing_sig.html")


# ## changing mu
# k = 2
# sig = 1
# mu = np.linspace(-3, 3, 13)
# data3 = {'phi': np.array([]), 'real': np.array([]), 'imaginary': np.array([]), 'mu': np.array([])}
# for i, mu_i in enumerate(mu):
#     mu_arr = np.array([mu_i]*len(x))
#     psi = np.multiply(np.exp(1j*k*x), np.sqrt(np.exp(-np.power(x-mu_i, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))))
#     y = np.real(psi)
#     z = np.imag(psi)
#     data3['phi'] = np.append(data3['phi'], x)
#     data3['real'] = np.append(data3['real'], y)
#     data3['imaginary'] = np.append(data3['imaginary'], z)
#     data3['mu'] = np.append(data3['mu'], mu_arr)

# data3['trace'] = ["wavefunction" for i in range(len(data3['phi']))]

# for i, mu_i in enumerate(mu):
#     mu_arr = np.array([mu_i]*len(x))
#     y = np.exp(-np.power(x-mu_i, 2)/(2*sig**2))/(sig*np.sqrt(2*np.pi))
#     z = -np.ones(x.shape)
#     data3['phi'] = np.append(data3['phi'], x)
#     data3['real'] = np.append(data3['real'], y)
#     data3['imaginary'] = np.append(data3['imaginary'], z)
#     data3['mu'] = np.append(data3['mu'], mu_arr)

# data3['trace'] = data3['trace'] + ["pdf" for i in range(len(data3['phi']) - len(data3['trace']))]
# df3 = pd.DataFrame(data = data3)
# fig3 = px.line_3d(df3, x="phi", y="real", z="imaginary", animation_frame="mu", color="trace")
# fig3["layout"].pop("updatemenus") # optional, drop animation buttons
# fig3.write_html("wavefunction_changing_mu.html")