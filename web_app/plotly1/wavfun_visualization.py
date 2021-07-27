import dash
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = dash.Dash()
func_map = {'exp(-x^2/k)': 1, 'exp(ikx)': 2, 'exp|ikx|': 3, 'cos(kx)': 4, 'x*exp(kx)': 5, None: 1}
app.layout = html.Div([
                 html.Div([html.P("Function:"), dcc.Dropdown(id='function-select', options=[{'label': '𝜓(x) = exp(−((x−x₀)/𝜅)^2−𝑖𝜙)', 'value': 'exp(-x^2/k)'}, 
                           {'label': '𝜓(x) = exp(𝑖𝜅(x−x₀)−𝑖𝜙)', 'value': 'exp(ikx)'}, {'label': '𝜓(x) = exp(𝑖𝜅∣x−x₀∣−𝑖𝜙)', 'value': 'exp|ikx|'},
                           {'label': '𝜓(x) = cos(𝜅(x−x₀))∗exp(−𝑖𝜙)', 'value': 'cos(kx)'}, {'label': '𝜓(x) = x∗exp(𝑖𝜅(x−x₀)−𝑖𝜙)', 'value': 'x*exp(kx)'}], 
                            value='exp(-x^2/k)')],
                           style={'width': '20%', 'font-size': '16px','display': 'inline-block', 'marginTop': '0px'}),
                 html.Div([html.P("x₀:"), dcc.Slider(id='x0-slider', included=False,
                           min=-2, max=2, step=0.01, marks={-2: {'label':'-2','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                   -1: {'label':'-1','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    0: {'label':'0','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    1: {'label':'1','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    2: {'label':'2','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}}, value=0)],
                           style={'width': '20%', 'font-size': '18px', 'display': 'inline-block','marginLeft': '20px'}),
                 html.Div([html.P("𝜅:"), dcc.Slider(id='k-slider', included=False,
                           min=-5, max=5, step=0.01, marks={-5: {'label':'-5','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    -3: {'label':'-3','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    -1: {'label':'-1','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     1: {'label':'1','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     3: {'label':'3','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     5: {'label':'5','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}}, value=3)], 
                           style={'width': '20%', 'font-size': '18px', 'display': 'inline-block'}),
                 html.Div([html.P("𝜙:"), dcc.Slider(id='phi-slider', included=False,
                           min=-np.pi, max=np.pi, step=0.01, marks={-np.pi: {'label':'-π','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    -0.5*np.pi: {'label':'-π/2','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     0: {'label':'0','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     0.5*np.pi: {'label':'π/2','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     np.pi: {'label':'π','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}}, value=0)], 
                          style={'width': '20%', 'font-size': '16px', 'display': 'inline-block'}),
                 html.Div([html.P("x-position (polar plot):"), dcc.Slider(id='x-slider', included=False, vertical = False,
                           min=-5, max=5, step=0.01, marks={-5: {'label':'-5','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    -3: {'label':'-3','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                    -1: {'label':'-1','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     1: {'label':'1','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     3: {'label':'3','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}, 
                                                     5: {'label':'5','style':{'color': 'black', 'font-size': '16px','font-family':'Times'}}}, value=3)], 
                           style={'width': '18%', 'font-size':'16px','display':'inline-block','position':'relative'}),
                 dcc.Graph(id='wvfunc-graph', config={"displayModeBar": False})])
                 
@app.callback(dash.dependencies.Output('wvfunc-graph', 'figure'),
    [dash.dependencies.Input('function-select', 'value'),
     dash.dependencies.Input('x0-slider', 'value'),
     dash.dependencies.Input('k-slider', 'value'),
     dash.dependencies.Input('phi-slider', 'value'),
     dash.dependencies.Input('x-slider', 'value'),])

def update_graph(func_name, x0, k, phi, xpos):
    fig = make_subplots(rows=2,cols=2,specs=[[{"rowspan": 2,"type":"scene"},{"type":"xy"}],[None,{"type":"polar"}]],column_widths=[0.7,0.4],row_heights=[0.5,0.5],horizontal_spacing = 0.01)
    drop_down = func_map[func_name]
    if drop_down == 1:
        xx = np.arange(-10, 10.05, 0.02)
        ref0 = np.zeros(len(xx))
        x_norm = np.arange(-50, 50, 0.01)
        psi_unnorm = np.round(np.exp(-(x_norm-x0)**2/k**2),3)*np.round(np.exp(-1j*phi),2)
        P_unnorm = np.conjugate(psi_unnorm) * psi_unnorm
        const = np.trapz(P_unnorm,x_norm)
        psi = np.round(np.exp(-(xx-x0)**2/k**2),3)*np.round(np.exp(-1j*phi),2)/np.sqrt(const)
        P_norm = np.real(np.conjugate(psi) * psi)
        psi_polar = np.round(np.exp(-(xpos-x0)**2/k**2),3)*np.round(np.exp(-1j*phi),2)/np.sqrt(const)
    elif drop_down == 2:
        xx = np.arange(-5, 5.05, 0.02)
        ref0 = np.zeros(len(xx))
        psi = np.round(np.exp(1j*k*(xx-x0)),3)*np.round(np.exp(-1j*phi),2)
        psi_polar = np.round(np.exp(1j*k*(xpos-x0)),3)*np.round(np.exp(-1j*phi),2)
    elif drop_down == 3:
        xx = np.arange(-5, 5.05, 0.02)
        ref0 = np.zeros(len(xx))
        psi = np.round(np.exp(1j*k*np.abs(xx-x0)),3)*np.round(np.exp(-1j*phi),2)
        psi_polar = np.round(np.exp(1j*k*np.abs(xpos-x0)),3)*np.round(np.exp(-1j*phi),2)
    elif drop_down == 4:
        xx = np.arange(-5, 5.05, 0.02)
        ref0 = np.zeros(len(xx))
        psi = np.round(np.cos(k*(xx-x0)),3)*np.round(np.exp(-1j*phi),2)
        psi_polar = np.round(np.cos(k*(xpos-x0)),3)*np.round(np.exp(-1j*phi),2)
    elif drop_down == 5:
        xx = np.arange(-5, 5.05, 0.02)
        ref0 = np.zeros(len(xx))
        psi = xx*np.round(np.exp(1j*k*(xx-x0)),3)*np.round(np.exp(-1j*phi),2)
        psi_polar = xpos*np.round(np.exp(1j*k*(xpos-x0)),3)*np.round(np.exp(-1j*phi),2)
    
    #3D_plot
    fig.add_trace(go.Scatter3d(line=dict(color="black", width=1), x=xx, y=ref0, z=ref0 ,name="x-axis", mode="lines"))                            
    fig.update_layout(scene = dict(xaxis_title='x',yaxis_title='Imag(𝜓)',zaxis_title='Real(𝜓)'))
    fig.add_trace(go.Scatter3d(line=dict(color="blue",width=1),x=xx,y=np.imag(psi),z=np.real(psi),name="𝜓(x)",mode="markers",
                               marker=dict(size=3,sizemode='diameter'), showlegend = False),row=1, col=1)
    fig.update_layout(scene_aspectmode='manual',scene_aspectratio=dict(x=2.25, y=0.9, z=0.9))
    fig.update_layout(scene_camera = dict(up=dict(x=0, y=0, z=1),center=dict(x=0, y=0, z=0),eye=dict(x=1.1, y=2.1, z=1.2)))
    fig.add_trace(go.Scatter3d(line=dict(color="#9c179e",width=3),x=[0,xpos],y=[0,np.imag(psi_polar)],z=[0,np.real(psi_polar)],
                    mode="lines", showlegend = False),row=1, col=1)
    fig.add_trace(go.Scatter3d(line=dict(color="#9c179e",width=3),x=[xpos],y=[np.imag(psi_polar)],z=[np.real(psi_polar)],
                    mode="markers",marker=dict(size=9,sizemode='diameter'), showlegend = False),row=1, col=1)
    #2D_plot-top
    fig.add_trace(go.Scatter(line=dict(color="#0d0887", width=1),x=xx,y=np.real(psi),name="Real(𝜓)"),row=1, col=2)
    fig.add_trace(go.Scatter(line=dict(color="red", width=1),x=xx,y=np.imag(psi),name="Imag(𝜓)"),row=1, col=2)
    fig.add_trace(go.Scatter(visible="legendonly",line=dict(color="green", width=1),x=xx,y=np.abs(psi),name="Abs(𝜓)"), row=1, col=2)
    fig.add_trace(go.Scatter(visible="legendonly",line=dict(color="black", width=1),x=xx,y=np.angle(psi),name="Angle(𝜓)"), row=1, col=2)
    #2D_plot-bottom
    fig.add_trace(go.Scatterpolar(mode='lines',r=[0,np.abs(psi_polar)],theta=[0,np.angle(psi_polar,deg=True)],name="𝜓(x)",
                                  line=dict(color="#9c179e", width=2), showlegend = False),row=2, col=2)
    fig.add_trace(go.Scatterpolar(mode='markers',r=[np.abs(psi_polar)],theta=[np.angle(psi_polar,deg=True)],name="𝜓(x)",
                                  marker=dict(color="#9c179e",size=8,sizemode='diameter'), showlegend = False),row=2, col=2)
    # fig.update_layout(legend=dict(yanchor="top", y=0.9, xanchor="right", x=0.9))
    
    #Figure_Axis-Setting
    fig.update_layout(autosize=False,width=1260,height=460,margin=dict(t=0, b=2, l=2, r=2), paper_bgcolor="white",hovermode='x')
    fig.update_layout(scene=dict(xaxis=dict(autorange='reversed')))
    fig.update_xaxes(range=[np.min(xx), np.max(xx)], dtick=1)
    fig.update_layout(legend_font_family="Times New Roman",font_size=14)
    zmax = np.ceil(np.max(np.abs(psi))*10)/10
    vals = np.around(np.linspace(-zmax,zmax,4),decimals=1)
    fig.update_layout(scene=dict(zaxis=dict(tickvals=vals,range=[-zmax,zmax]),yaxis=dict(tickvals=vals,range=[-zmax,zmax])))
    #2D_plot-top
    fig.update_xaxes(title='x', row=1, col=2)
    xmin, xmax = xx[0], xx[len(xx)-1]
    xvals = np.around(np.linspace(xmin,xmax,5),decimals=1)
    fig.update_xaxes(range=[xmin, xmax], tickvals=xvals, dtick=1, row=1, col=2)
    #2D_plot-bottom
    fig.update_polars(radialaxis_nticks=4)
    # rmin, rmax = 0, np.ceil(np.abs(psi_polar)*10)/10
    # rvals = np.around(np.linspace(rmin,rmax,4),decimals=1)
    # fig.update_polars(radialaxis_range=[rmin, rmax], radialaxis_tickvals=rvals)
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
    