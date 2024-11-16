import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import numpy as np
from statsmodels.graphics.tsaplots import pacf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt
import numpy as np
from skopt import BayesSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import altair as alt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from IPython.display import display
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import jarque_bera
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
import mglearn
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Tuple, List, Any
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# Librerías y datos
from jupyter_dash import JupyterDash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np

from dash import html, dcc
import dash_bootstrap_components as dbc

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY]) 
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
values = np.random.randn(100).cumsum()
subjects = np.random.choice(['Sujeto 1', 'Sujeto 2', 'Sujeto 3','Sujeto 4','Sujeto 5'], size=100)  # Sujetos aleatorios
exercises = np.random.choice(['Ejercicio 1', 'Ejercicio 2','Ejercicio 3','Ejercicio 4','Ejercicio 5','Ejercicio 6','Ejercicio 7','Ejercicio 8'], size=100)
units = np.random.choice(['Unidad 1', 'Unidad 2', 'Unidad 3', 'Unidad 4', 'Unidad 5'], size=100)

df = pd.DataFrame({
    "Date": dates, 
    "Value": values, 
    "Sujeto": subjects, 
    "Ejercicio": exercises,
    "Unidad": units
})
dx = {
    'Sujeto 1':'s1', 'Sujeto 2':'s2', 'Sujeto 3':'s3','Sujeto 4':'s4','Sujeto 5':'s5',
    'Ejercicio 1':   'e1',
    'Ejercicio 2':   'e2',
    'Ejercicio 3':   'e3',
    'Ejercicio 4':   'e4',
    'Ejercicio 5':   'e5',
    'Ejercicio 6':   'e6',
    'Ejercicio 7':   'e7',
    'Ejercicio 8':   'e8',
    'Unidad 1':'u1', 'Unidad 2':'u2', 'Unidad 3':'u3', 'Unidad 4':'u4', 'Unidad 5':'u5',
            'acc':'Accelerometer',
            'gyr':'Gyroscope',
            'mag':'Magnetometer'
}
app.layout = dbc.Container(
    [
    # Título
    dbc.Row(
        dbc.Col(
            html.H2(
                "Seleccione un Sujeto, Ejercicio y Unidad",
                className="text-center text-light mt-4 mb-4"  # Cambiado a `text-light` para fondo oscuro
            )
        )
    ),
    # Opciones de selección
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Label("Sujeto", className="fw-bold text-light"),  # Cambiado a `text-light`
                    dcc.RadioItems(
                        id="subject-radio",
                        options=[
                            {'label': f"🔹 {s}", 'value': s} for s in ["Sujeto 1", "Sujeto 2", "Sujeto 3", "Sujeto 4", "Sujeto 5"]
                        ],
                        value="Sujeto 1",  # Valor inicial
                        labelStyle={
                            'display': 'block',
                            'padding': '10px',
                            'border': '1px solid #ddd',
                            'border-radius': '5px',
                            'margin-bottom': '5px',
                            'background-color': '#212529',  # Fondo oscuro
                            'color': '#f8f9fa',  # Texto claro
                            'cursor': 'pointer',
                            'font-size': '16px'
                        },
                        inputStyle={"margin-right": "10px"}
                    ),
                ],
                width=4
            ),
            dbc.Col(
                [
                    html.Label("Ejercicio", className="fw-bold text-light"),  # Cambiado a `text-light`
                    dcc.RadioItems(
                        id="exercise-radio",
                        options=[
                            {'label': f"⚡ {e}", 'value': e} for e in [
                                "Ejercicio 1", "Ejercicio 2", "Ejercicio 3", "Ejercicio 4",
                                "Ejercicio 5", "Ejercicio 6", "Ejercicio 7", "Ejercicio 8"
                            ]
                        ],
                        value="Ejercicio 1",  # Valor inicial
                        labelStyle={
                            'display': 'block',
                            'padding': '10px',
                            'border': '1px solid #ddd',
                            'border-radius': '5px',
                            'margin-bottom': '5px',
                            'background-color': '#212529',  # Fondo oscuro
                            'color': '#f8f9fa',  # Texto claro
                            'cursor': 'pointer',
                            'font-size': '16px'
                        },
                        inputStyle={"margin-right": "10px"}
                    ),
                ],
                width=4
            ),
            dbc.Col(
                [
                    html.Label("Unidad", className="fw-bold text-light"),  # Cambiado a `text-light`
                    dcc.RadioItems(
                        id="unit-radio",
                        options=[
                            {'label': f"📦 {u}", 'value': u} for u in [
                                "Unidad 1", "Unidad 2", "Unidad 3", "Unidad 4", "Unidad 5"
                            ]
                        ],
                        value="Unidad 1",  # Valor inicial
                        labelStyle={
                            'display': 'block',
                            'padding': '10px',
                            'border': '1px solid #ddd',
                            'border-radius': '5px',
                            'margin-bottom': '5px',
                            'background-color': '#212529',  # Fondo oscuro
                            'color': '#f8f9fa',  # Texto claro
                            'cursor': 'pointer',
                            'font-size': '16px'
                        },
                        inputStyle={"margin-right": "10px"}
                    ),
                ],
                width=4
            ),
        ]
    ),
    # Gráfica
    dbc.Row(
        dbc.Col(
            dcc.Graph(id="table_0"),
            width=12
        )
    ),
       # Descripción
dbc.Row(
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Descripción", className="card-title text-primary"),
                    html.P(
                        """
                        La tabla resume las estadísticas descriptivas de las mediciones para el Sujeto 3 en el Ejercicio 1, Unidad 1. 
                        Las variables de aceleración (`acc_x`, `acc_y`, `acc_z`) oscilan entre aproximadamente -10 y 1.5, mientras que las magnitudes (`mag_x`, `mag_y`, `mag_z`) permanecen en valores positivos, con un rango de 0.1 a 0.7. 
                        Estos datos reflejan la variabilidad de las mediciones en el tiempo.
                        """,
                        className="text-muted"
                    )
                ]
            ),
            className="mt-3 shadow-sm"
        ),
        width=12
    )
),
    # Título de la sección
    dbc.Row(
        dbc.Col(
            html.H2(
                "Series Temporales",
                className="text-light text-center mt-4 mb-3"
            )
        )
    ),
    # Subtítulo
    dbc.Row(
        dbc.Col(
            html.H3(
                "Seleccione entre Accelerometer, Gyroscope o Magnetometer.",
                className="text-light text-center mb-4"
            )
        )
    ),
# Dropdown estilizado con Dash
dbc.Row(
    dbc.Col(
        dcc.Dropdown(
            id='ac-gr-mg',
            options=[
                {'label': 'Accelerometer', 'value': 'acc'},
                {'label': 'Gyroscope', 'value': 'gyr'},
                {'label': 'Magnetometer', 'value': 'mag'}
            ],
            value='acc',  # Valor inicial
            clearable=False,
            style={
                'backgroundColor': '#FFFFFF',  # Fondo gris oscuro del dropdown
                'color': '#FF0000',  # Texto blanco en el campo seleccionado
                'border': '1px solid #6c757d',  # Borde gris claro
                'borderRadius': '5px',  # Bordes redondeados
                'fontSize': '16px',  # Tamaño de fuente
                'padding': '10px'  # Espaciado interno
            },
        ),
        width=6,  # Ajusta el ancho
        className="mx-auto"  # Centra el dropdown
    )
),
html.Br(), html.Br(), html.Br(),
dcc.Graph(id="fig_1"),
dcc.Graph(id="fig_2"),
dcc.Graph(id="fig_3"),
dbc.Row(
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Descripción", className="card-title text-primary"),
                    html.P(
                        """
                        Las series temporales muestran los datos registrados por el giroscopio en las tres dimensiones (`x`, `y`, `z`):

                        Los valores oscilan entre aproximadamente -valores negativos y positivos, con fluctuaciones constantes y una alta densidad de picos distribuidos de manera uniforme.
                        """,
                        className="text-muted"
                    )
                ]
            ),
            className="mt-3 shadow-sm"  # Margen superior y ligera sombra
        ),
        width=12  # Ancho completo de la columna
    )
),
html.H2("Gráficos de Cajas", className="text-light text-center mt-4"),
dcc.Graph(id="fig_4"),
dbc.Row(
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Descripción", className="card-title text-primary"),
                    html.P(
                        """
                        El gráfico muestra la distribución de los valores registrados por el acelerómetro en tres dimensiones:

                        1. En la primera dimensión tenemos la x, los valores están centrados alrededor de posiciones negativas, con algunos valores atípicos hacia el extremo superior.
                        2. En la segunda dimensión tenemos la y, con una distribución más alargada hacia valores negativos y varios valores atípicos hacia el extremo positivo.
                        3. En la tercera dimensión tenemos la z, los valores son predominantemente positivos, con una mediana apreciada y algunos valores atípicos.

                        Este análisis resalta diferencias clave en las distribuciones y la presencia de valores atípicos en cada dimensión.
                        """,
                        className="text-muted"
                    )
                ]
            ),
            className="mt-3 shadow-sm"
        ),
        width=12
    )
),

html.H2("Descomposición estacional aditiva", className="text-light text-center mt-4"),
html.Br(), html.Br(), html.Br(), html.Br(),
dcc.Graph(id="fig_5"),
dbc.Row(
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Descripción", className="card-title text-primary"),
                    html.P(
                        """
                        El gráfico presenta una descomposición aditiva de una serie temporal, separando sus componentes principales: la serie observada, la tendencia, la estacionalidad y los residuos. 
                        Esta visualización permite identificar patrones subyacentes como fluctuaciones regulares (estacionalidad), 
                        cambios a largo plazo (tendencia) y variaciones no explicadas por los componentes principales (residuos). 
                        Es útil para analizar cómo se combinan estos elementos para formar la serie observada.
                        """,
                        className="text-muted"
                    )
                ]
            ),
            className="mt-3 shadow-sm"
        ),
        width=12
    )
),

# Sección: Selección de variables para Autocorrelación Parcial
dbc.Row(
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    # Título Principal
                    html.H2(
                        "Seleccione las variables para la Autocorrelación Parcial",
                        className="text-primary text-center mb-4"
                    ),
                    # Selección de Sujeto
                    html.Label("Seleccione un Sujeto", className="text-light fw-bold"),
                    dcc.RadioItems(
                        id="subject-radio2",
                        options=[{'label': s, 'value': s} for s in df["Sujeto"].unique()],
                        value='Sujeto 1',
                        labelStyle={
                            'display': 'block',
                            'padding': '10px',
                            'cursor': 'pointer',
                            'border-radius': '5px',
                            'background-color': '#343a40',  # Fondo oscuro
                            'color': '#FFFFFF'  # Texto blanco
                        },
                        inputStyle={"margin-right": "10px"}
                    ),
                    html.Br(),
                    # Selección de Ejercicio
                    html.Label("Seleccione un Ejercicio", className="text-light fw-bold"),
                    dcc.RadioItems(
                        id="exercise-radio2",
                        options=[{'label': e, 'value': e} for e in df["Ejercicio"].unique()],
                        value='Ejercicio 1',
                        labelStyle={
                            'display': 'block',
                            'padding': '10px',
                            'cursor': 'pointer',
                            'border-radius': '5px',
                            'background-color': '#343a40',  # Fondo oscuro
                            'color': '#FFFFFF'  # Texto blanco
                        },
                        inputStyle={"margin-right": "10px"}
                    ),
                    html.Br(),
                    # Selección de Tipo de Examen
                    html.Label("Seleccione el Tipo de Examen", className="text-light fw-bold"),
                    dcc.Dropdown(
                        id='ac-gr-mg2',
                        options=[
                            {'label': 'Accelerometer', 'value': 'acc'},
                            {'label': 'Gyroscope', 'value': 'gyr'},
                            {'label': 'Magnetometer', 'value': 'mag'}
                        ],
                        value='acc',
                        clearable=False,
                        style={
                            'backgroundColor': '#FFFFFF',  # Fondo gris oscuro del dropdown
                            'color': '#FF0000',  # Texto blanco
                            'border-radius': '5px',
                            'border': '1px solid #6c757d',  # Borde gris claro
                            'padding': '10px'
                        }
                    ),
                    html.Br(),
                    # Selección de Columna
                    html.Label("Seleccione la Columna (x, y, z)", className="text-light fw-bold"),
                    dcc.Dropdown(
                        id='x-y-z',
                        options=[
                            {'label': 'x', 'value': 'x'},
                            {'label': 'y', 'value': 'y'},
                            {'label': 'z', 'value': 'z'}
                        ],
                        value='x',
                        clearable=False,
                        style={
                            'backgroundColor': '#FFFFFF',  # Fondo gris oscuro del dropdown
                            'color': '#008000',  # Texto blanco
                            'border-radius': '5px',
                            'border': '1px solid #6c757d',  # Borde gris claro
                            'padding': '10px'
                        }
                    )
                ]
            ),
            className="mt-4 shadow-sm"
        ),
        width=12
    )
),
html.Br(), html.Br(), html.Br(), html.Br(),
dcc.Graph(id="fig_6"),
dbc.Row(
    dbc.Col(
        dbc.Card(
            dbc.CardBody(
                [
                    html.H4("Descripción", className="card-title text-primary"),
                    html.P(
                        """
                        Las gráficas muestran la autocorrelación parcial (PACF) para distintas unidades de datos.
                        En general, se observa que en los primeros retrasos (lags), algunas barras superan las bandas de significancia, 
                        lo que indica que existe una correlación significativa en esos retrasos. A medida que aumentan los retrasos, 
                        las correlaciones se acercan a cero y permanecen dentro de las bandas, sugiriendo que la influencia de los valores anteriores disminuye con el tiempo.
                        
                        Esto es típico de series temporales con dependencias a corto plazo. La presencia de valores significativos iniciales podría ser relevante para ajustar 
                        modelos autoregresivos (AR) en estas series.
                        """,
                        className="text-muted"
                    )
                ]
            ),
            className="mt-3 shadow-sm"
        ),
        width=12
    )
)
    
])

@app.callback(
    Output("table_0", "figure"),
    Output("fig_1", "figure"),
    Output("fig_2", "figure"),
    Output("fig_3", "figure"),
    Output("fig_4", "figure"),
    Output("fig_5", "figure"),
    Output("fig_6", "figure"),
    [Input("subject-radio", "value"),
     Input("exercise-radio", "value"),
     Input("unit-radio", "value"),
     Input('ac-gr-mg', "value"),
     Input("subject-radio2", "value"),
     Input("exercise-radio2", "value"),
     Input('ac-gr-mg2', "value"),
     Input('x-y-z', "value")])

def update_chart(selected_subject, selected_exercise, selected_unit, type_df, selected_subject2, selected_exercise2, type_df2, xyz):
    path = f"fisioterapia_dataset_regresion/{dx[selected_subject]}/{dx[selected_exercise]}/{dx[selected_unit]}/template_session.txt"
    data_df = pd.read_csv(path, delimiter=';')  
    filtered_df = data_df
    data_df = data_df.reset_index(drop=True)
    len_X = [n for n in range(len(data_df['time index']))]
    fig_0 = go.Figure()
    fig_0.add_trace(go.Scatter(x=len_X, y=data_df[f"{type_df}_x"], mode='lines', name=f"{type_df}_x", 
        line=dict(color="red")))
    fig_0.update_layout(title=f"Serie Temporal {dx[type_df]} x")
    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=len_X, y=data_df[f"{type_df}_y"], mode='lines', name=f"{type_df}_y", 
                    line=dict(color="green")))
    fig_1.update_layout(title=f"Serie Temporal {dx[type_df]} y")
    fig_2 = go.Figure()
    fig_2.add_trace(go.Scatter(x=len_X, y=data_df[f"{type_df}_z"], mode='lines', name=f"{type_df}_z", 
        line=dict(color="purple")))
    fig_2.update_layout(title=f"Serie Temporal {dx[type_df]} z")
    table_0 = go.Figure()
    describe_df = filtered_df.describe()
    table_data = describe_df.reset_index()
    table = go.Table(
        header=dict(values=["Estadísticas"] + list(table_data.columns[1:])),
        cells=dict(values=[table_data["index"]] + [table_data[col] for col in table_data.columns[1:]])
    )
    table_0.add_trace(table)
    table_0.update_layout(
        title=f"Tabla Descriptiva - {selected_subject}, {selected_exercise}, {selected_unit}",
        xaxis_title="Fecha",
        yaxis_title="Valor" )
    boxes = make_subplots(rows=1, cols=3, shared_xaxes=True, vertical_spacing=0.1)
    boxes.add_trace(go.Box(y=data_df[f"{type_df}_x"], name=f'{type_df}_x', marker_color='red'), row=1, col=1)
    boxes.add_trace(go.Box(y=data_df[f"{type_df}_y"], name=f'{type_df}_y', marker_color='green'), row=1, col=2)
    boxes.add_trace(go.Box(y=data_df[f"{type_df}_z"], name=f'{type_df}_z', marker_color='purple'), row=1, col=3)

    boxes.update_layout(
        title = f'Box Plot de {dx[type_df]}',
        xaxis_title = 'Time Index',
        yaxis_title = 'Valor',
        boxmode = 'group',
        height = 900  
    )
    df_plot = data_df.sort_values(by='time index')
    df_plot = df_plot.set_index('time index')
    type_df = 'gyr'
    result_add = seasonal_decompose(df_plot[f"{type_df}_x"], model='additive', period=24*7)
    fig_3 = make_subplots(rows=1, cols=1, shared_xaxes=True, subplot_titles=["Additive Decomposition"], vertical_spacing=0.1)
    def add_decomposition(result, row):
        fig_3.add_trace(go.Scatter(x=result.observed.index, y=result.observed, mode='lines', name='Observed', line=dict(color='blue')), row=row, col=1)
        fig_3.add_trace(go.Scatter(x=result.trend.index, y=result.trend, mode='lines', name='Trend', line=dict(color='orange')), row=row, col=1)
        fig_3.add_trace(go.Scatter(x=result.seasonal.index, y=result.seasonal, mode='lines', name='Seasonal', line=dict(color='green')), row=row, col=1)
        fig_3.add_trace(go.Scatter(x=result.resid.index, y=result.resid, mode='lines', name='Residual', line=dict(color='red')), row=row, col=1)
    add_decomposition(result_add, row=1)
    fig_3.update_layout(title_text=f'Descomposición estacional de {type_df}_x (Aditiva)', height=800, width=1200, showlegend=True)
    '''    for unit in range(1,5):
        temp_path = rf"C://Users//loren//OneDrive\Escritorio//fisioterapia_dataset_regresion//{dx[selected_subject2]}//{dx[selected_exercise2]}//u{unit}//template_session.txt" 
        actual = pd.read_csv(temp_path, delimiter=';')
        hour_data = actual[f'{type_df}_{xyz}'].diff().dropna()
        plot_pacf(hour_data, lags=30, alpha=0.01)
        plt.title(f'Autocorrelacion Parcial - Unidad {unit}')
        plt.ylabel('Correlation')
        plt.xlabel('Lags')
        plt.show()'''
    lags=30;alpha=0.01
    fig_4=make_subplots(rows=5, cols=1, subplot_titles=[f'Autocorrelación Parcial - Unidad {i+1} {type_df2}_{xyz}' for i in range(5)], vertical_spacing=0.1)
    for unit in range(1, 6):
        temp_path=f"fisioterapia_dataset_regresion/{dx[selected_subject2]}/{dx[selected_exercise2]}/u{unit}/template_session.txt"
        actual=pd.read_csv(temp_path, delimiter=';')
        hour_data=actual[f'{type_df2}_{xyz}'].diff().dropna()
        pacf_values=pacf(hour_data, nlags=lags, alpha=alpha)
        pacf_y=pacf_values[0]
        conf_int=pacf_values[1]
        lower_conf=conf_int[:,0]-pacf_y
        upper_conf=conf_int[:,1]-pacf_y

        fig_4.add_trace(go.Bar(x=list(range(len(pacf_y))), y=pacf_y, marker_color='blue', name=f'PACF Unidad {unit}'), row=unit, col=1)
        fig_4.add_trace(go.Scatter(x=list(range(len(pacf_y))), y=upper_conf, mode='lines', line=dict(color='red', dash='dash'), name='Límite superior'), row=unit, col=1)
        fig_4.add_trace(go.Scatter(x=list(range(len(pacf_y))), y=lower_conf, mode='lines', line=dict(color='red', dash='dash'), name='Límite inferior'), row=unit, col=1)

    fig_4.update_layout(height=800, width=1200, title_text='Autocorrelación Parcial (PACF) por tipo de examen y columna', showlegend=False)    
    return table_0, fig_0, fig_1, fig_2, boxes, fig_3, fig_4
server = app.server
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(debug=False, host="0.0.0.0", port=port)