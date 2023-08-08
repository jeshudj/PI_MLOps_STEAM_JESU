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
@app.get("/genero/")
def genero(año: str):
    año_filtro = data[data['release_year'] == int(año)] 
    generos_count = año_filtro['genres'].value_counts().head(5)
    generos_dict = generos_count.to_dict()
    return {'Año': año, 'Cantidad de géneros': generos_dict}

# CONSULTA 2:
# def juegos( Año: str ): Se ingresa un año y devuelve una lista con los juegos lanzados en el año.
@app.get("/juegos/")
def juegos(año: str):
    juegos_año = data[data['release_year'] == int(año)]['app_name'].head(10).tolist()
    return {'Año': año, 'Juegos lanzados': juegos_año}


# CONSULTA 3:
# def specs( Año: str ): Se ingresa un año y devuelve una lista con los 5 specs que más 
# se repiten en el mismo en el orden correspondiente.
@app.get("/specs/")
def specs(año: str):
    año_filtro = data[data['release_year'] == int(año)] 
    specs_count = año_filtro['specs'].value_counts().head(5)
    specs_dict = specs_count.to_dict()
    return {'Año': año, 'Cantidad de especificaciones': specs_dict}



# CONSULTA 4:
# def earlyacces( Año: str ): Cantidad de juegos lanzados en un año con early access.
@app.get("/early_access/")
def early_access(año: str):
    juegos_early_access = data[(data['release_year'] == int(año)) & (data['early_access'] == True)]
    cantidad_juegos_early_access = juegos_early_access.shape[0]
    return {'Año': año, 'Cantidad de juegos con Early Access': cantidad_juegos_early_access}


# CONSULTA 5:
# Función para obtener el análisis de sentimiento por año
@app.get("/sentiment/")
def sentiment(año: str):
    registros_sentimiento = data[data['release_year'] == int(año)]['sentiment'].value_counts().to_dict()
    return {'Año': año, 'Registros de Sentimiento': registros_sentimiento}

# CONSULTA 6:
# def metascore( Año: str ): Top 5 juegos según año con mayor metascore.
@app.get("/metascore/")
def metascore(año: str):
    
    data['metascore'] = pd.to_numeric(data['metascore'], errors='coerce')
        
    juegos_metascore = data[data['release_year'] == int(año)].nlargest(5, 'metascore')[['app_name', 'metascore']]
       
    juegos_metascore = juegos_metascore.set_index('app_name').to_dict()['metascore']
    
    return {'Año': año, 'Top 5 juegos con mayor Metascore': juegos_metascore}





# MODELO DE PREDICIION



