### IMPORTAMOS LIBRERIAS

import pandas as pd
import numpy  as np
from fastapi import FastAPI
import uvicorn
from sklearn.metrics.pairwise        import cosine_similarity
from sklearn.utils.extmath           import randomized_svd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import linear_kernel
from sklearn.neighbors               import NearestNeighbors

app = FastAPI()

# Creaoms una consulta como presentacion con nuestro nombre
@app.get('/')
def presentacion():
    return 'Jesus Marceliano'

# IMPORTAMOS LOS DATOS
data = pd.read_csv("clean_games.csv", sep=';', encoding='utf-8')


# CONSULTA 1:
# def genero( Año: str ): Se ingresa un año y devuelve una lista con los 5 géneros 
# más ofrecidos en el orden correspondiente.
@app.get("/genero/{genero}")
def genero(año: str):
    año_filtro = data[data['release_year'] == año]
    top_generos = año_filtro['genres'].value_counts().head(5).index.tolist()
    return {'Año': año, 'Top 5 géneros': top_generos}

# CONSULTA 2:
# def juegos( Año: str ): Se ingresa un año y devuelve una lista con los juegos lanzados en el año.
@app.get("/juegos/{juegos}")
def juegos(año: str):
    juegos_año = data[data['release_year'] == año]['app_name'].tolist()
    return {'Año': año, 'Juegos lanzados': juegos_año}


# CONSULTA 3:
# def specs( Año: str ): Se ingresa un año y devuelve una lista con los 5 specs que más 
# se repiten en el mismo en el orden correspondiente.
@app.get("/specs/{specs}")
def specs(año: str):
    specs_año = data[data['release_year'] == año]['specs'].value_counts().head(5).index.tolist()
    return {'Año': año, 'Top 5 specs': specs_año}



# CONSULTA 4:
# def earlyacces( Año: str ): Cantidad de juegos lanzados en un año con early access.
@app.get("/early_access/{early_access}")
def early_access(año: str):
    juegos_early_access = data[(data['release_year'] == año) & (data['early_access'] == True)]
    cantidad_juegos_early_access = juegos_early_access.shape[0]
    return {'Año': año, 'Cantidad de juegos con Early Access': cantidad_juegos_early_access}


# CONSULTA 5:
# Ingresas la productora, entregandote el revunue total y la cantidad de peliculas que realizo
@app.get("/sentiment/{sentiment}")
def sentiment(año: str):
    registros_sentimiento = data[data['release_year'] == año]['sentiment'].value_counts().to_dict()
    return {'Año': año, 'Registros de Sentimiento': registros_sentimiento}

# CONSULTA 6:
# def metascore( Año: str ): Top 5 juegos según año con mayor metascore.
@app.get("/metascore/{metascore}")
def metascore(año: str):
    juegos_metascore = data[data['release_year'] == año].nlargest(5, 'metascore')[['app_name', 'metascore']]
    juegos_metascore = juegos_metascore.set_index('app_name').to_dict()['metascore']
    return {'Año': año, 'Top 5 juegos con mayor Metascore': juegos_metascore}





# MODELO DE PREDICIION

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Separar las características y el objetivo
X = data[['genres', 'early_access', 'metascore', 'release_year']]
y = data['real_price']

# Codificar variables categóricas si es necesario

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular el RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Función en la API para calcular el precio y el RMSE
@app.get("/prediccion1")
def prediccion(genres: str, early_access: bool, metascore: float, release_year: int):
    # Utilizar el modelo entrenado para predecir el precio
    precio = modelo.predict([[genres, early_access, metascore, release_year]])

    return {'Precio': precio, 'RMSE': rmse}




import ipywidgets as widgets
from IPython.display import display

# Crear menús desplegables para cada característica
genres_dropdown = widgets.Dropdown(options=data['genres'].unique(), description='Genres:')
early_access_dropdown = widgets.Dropdown(options=[True, False], description='Early Access:')
metascore_dropdown = widgets.FloatSlider(min=0, max=100, step=0.1, description='Metascore:')
release_year_dropdown = widgets.Dropdown(options=data['release_year'].unique(), description='Release Year:')

# Función en la API para calcular el precio y el RMSE
@app.get("/prediccion2")
def prediccion():
    # Obtener los valores seleccionados de los menús desplegables
    genres = genres_dropdown.value
    early_access = early_access_dropdown.value
    metascore = metascore_dropdown.value
    release_year = release_year_dropdown.value

    # Utilizar el modelo entrenado para predecir el precio
    precio = modelo.predict([[genres, early_access, metascore, release_year]])

    return {'Precio': precio, 'RMSE': rmse}

# Mostrar los menús desplegables
display(genres_dropdown, early_access_dropdown, metascore_dropdown, release_year_dropdown)