import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

#Data
x = np.arange(-3, 3.05, 0.05)
ref0 = np.zeros(len(x))
psi1 = np.exp(-x**2)
psi2 = np.exp(1j*2*x)
psi3 = np.exp(-1j*2*x)
psi4 = np.sin(x) * np.exp(-1j*2*x)
psi5 = np.cos(x) * np.exp(-1j*2*x)
psi = np.vstack((psi1, psi2, psi3, psi4, psi5))
psi_list = ['exp(-x^2)', 'exp(2i*x)', 'exp(-2i*x)', 'sin(x)*exp(-2i*x)', 'cos(x)*exp(-2i*x)']
x_list = ['x = -3', 'x = -2.5','x = -2', 'x = -1.5','x = -1', 'x = -0.5','x = 0', 'x = 0.5','x = 1', 'x = 1.5','x = 2', 'x = 2.5','x = 3']

#Subplot_setting
fig = make_subplots(rows=2,cols=2,specs=[[{"rowspan": 2,"type":"scene"},{"type":"xy"}],[None,{"type":"polar"}]],column_widths=[0.7,0.4],row_heights=[0.5,0.6],horizontal_spacing = 0.01)

#3D_plot
fig.add_trace(go.Scatter3d(visible=True,line=dict(color="black", width=1), x=x, y=ref0, z=ref0 ,name="x", mode="lines"))                            #0
fig.update_layout(scene = dict(xaxis_title='x',yaxis_title='Imag(Ψ)',zaxis_title='Real(Ψ)'))
fig.update_layout(scene_camera = dict(up=dict(x=0, y=0, z=1),center=dict(x=0, y=0, z=0),eye=dict(x=1.1, y=1.7, z=1.25)))
for k1 in range(0, len(psi)):            # (1 - 5)
    fig.add_trace(go.Scatter3d(visible=False,line=dict(color="blue",width=1),x=x,y=np.imag(psi[k1]),z=np.real(psi[k1]),name="Ψ",mode="markers"),row=1,col=1) 
fig.data[1].visible = True

#2D_plot-top
for k1 in range(0, len(psi)):            # (6 - 9), (10 - 13), (14 - 17), (18 - 21), (22 - 25)
    fig.add_trace(go.Scatter(visible=False,line=dict(color="blue", width=1),x=x,y=np.real(psi[k1]),name="Real(Ψ)"), row=1, col=2)         #6
    fig.add_trace(go.Scatter(visible=False,line=dict(color="black", width=1),x=x,y=np.imag(psi[k1]),name="Imag(Ψ)"), row=1, col=2)        #7
    fig.add_trace(go.Scatter(visible=False,line=dict(color="red", width=1),x=x,y=np.abs(psi[k1]),name="Abs(Ψ)"), row=1, col=2)            #8
    fig.add_trace(go.Scatter(visible=False,line=dict(color="green", width=1),x=x,y=np.angle(psi[k1]),name="Angle(Ψ)"), row=1, col=2)      #9    
fig.data[6].visible = True
fig.data[7].visible = "legendonly"
fig.data[8].visible = "legendonly"
fig.data[9].visible = "legendonly"

#2D_plot-bottom
for k1 in range(0, len(psi)):            # (26 - 30)     
    fig.add_trace(go.Scatterpolar(visible=False,mode='markers',r=np.abs(psi[k1]),theta=np.angle(psi[k1],deg=True),name="Ψ", text=['x: ' + str(x_i) for x_i in x], 
    marker=dict(color=x, colorbar=dict(title="Colorbar"), colorscale="Viridis", showscale=False)), row=2, col=2)
fig.data[26].visible = True

#for k1 in range(27,39):
    #fig.data[k1].visible = "legendonly"

#Figure_Axis-Setting
fig.update_layout(title_text='Wavefunction Visualization',autosize=False,width=1270,height=550,margin=dict(t=35, b=10, l=10, r=10), paper_bgcolor="white")
fig.update_xaxes(range=[-3, 3], dtick=1)
fig.update_xaxes(title='x', row=1, col=2)
fig.update_yaxes(range=[-3, 3], dtick=1, row=1, col=2)
fig.update_layout(legend_font_family="Calibri",font_size=15)
#fig.update_layout(title='Phasor(Ψ)', row=2, col=2)

#Update_with_Dropdown_menus
steps = []
for k1 in range(0, len(psi)):
    step = dict(method="restyle", label="Ψ = " + psi_list[k1], args = [{"visible": [False] * (1+18*len(psi))}],)    # All set to False
    step["args"][0]["visible"][0] = True
    step["args"][0]["visible"][k1+1] = True
    step["args"][0]["visible"][4*k1+6] = True
    step["args"][0]["visible"][4*k1+7] = 'legendonly'
    step["args"][0]["visible"][4*k1+8] = 'legendonly'
    step["args"][0]["visible"][4*k1+9] = 'legendonly'
    #step["args"][0]["visible"][13*k1+26] = True    
    step["args"][0]["visible"][26+k1] = True    
    #step["args"][0]["visible"][13*k1+k2] = 'legendonly'
    steps.append(step)

fig.update_layout(updatemenus=[dict(buttons=list([
            dict(steps[0]),
            dict(steps[1]),
            dict(steps[2]),
            dict(steps[3]),
            dict(steps[4])]),
            direction="down", pad={"r": 10, "t": 10}, showactive=True,
            x=0, xanchor="left", y=1, yanchor="top"),])
fig.show()
fig.write_html("WV3D_sandbox.html")