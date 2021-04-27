# import plotly.graph_objects as go
# fig = go.Figure(data=go.Bar(y=[2, 3, 1]))
# fig.write_html('first_figure.html', auto_open=True)

# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# from dash.dependencies import Input, Output
# import plotly.graph_objects as go

# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.P("Color:"),
#     dcc.Dropdown(
#         id="dropdown",
#         options=[
#             {'label': x, 'value': x}
#             for x in ['Gold', 'MediumTurquoise', 'LightGreen']
#         ],
#         value='Gold',
#         clearable=False,
#     ),
#     dcc.Graph(id="graph"),
# ])

# @app.callback(
#     Output("graph", "figure"), 
#     [Input("dropdown", "value")])
# def display_color(color):
#     fig = go.Figure(
#         data=go.Bar(y=[2, 3, 1], marker_color=color))
#     return fig

# app.run_server(debug=True)\

import pandas as pd
import numpy as np
import plotly.express as px

rs = np.random.RandomState()
rs.seed(0)

# def brownian_motion(T = 1, N = 100, mu = 0.1, sigma = 0.01, S0 = 20):
#     dt = float(T)/N
#     t = np.linspace(0, T, N)
#     W = rs.standard_normal(size = N)
#     W = np.cumsum(W)*np.sqrt(dt) # standard brownian motion
#     X = (mu-0.5*sigma**2)*t + sigma*W
#     S = S0*np.exp(X) # geometric brownian motion
#     return S

# dates = pd.date_range('2012-01-01', '2013-02-22')
# T = (dates.max()-dates.min()).days / 365
# N = dates.size
# start_price = 100
# y = brownian_motion(T, N, sigma=0.1, S0=start_price)
# z = brownian_motion(T, N, sigma=0.1, S0=start_price)

# fig = px.line_3d(pd.DataFrame(data=dict(x=dates, y=y, z=z)), x="x", y="y", z="z")

# fig.update_layout(
#     width=800,
#     height=700,
#     autosize=False,
#     scene=dict(
#         camera=dict(
#             up=dict(
#                 x=0,
#                 y=0,
#                 z=1
#             ),
#             eye=dict(
#                 x=0,
#                 y=1.0707,
#                 z=1,
#             )
#         ),
#         aspectratio = dict( x=1, y=1, z=0.7 ),
#         aspectmode = 'manual'
#     ),
# )


k=3
phi_0 = 0
sig = 1
x = np.linspace(-5, 5, 150)
psi = np.multiply(np.exp(-1j*k*x), np.exp(-np.power(x-phi_0, 2)/(2*sig**2))/(2*sig*np.sqrt(np.pi)))
y = np.real(psi)
z = np.imag(psi)
df = pd.DataFrame(data=dict(x=x, y=y, z=z))
fig = px.line_3d(df, x="x", y="y", z="z")

fig.show()