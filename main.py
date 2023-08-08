### IMPORTAMOS LIBRERIAS

import pandas as pd
import numpy  as np
import uvicorn
import pickle
import ast
from enum                            import Enum
from fastapi                         import FastAPI
from fastapi                         import FastAPI, HTTPException
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
# Cargar el modelo pickle
with open("predic_jesu.pkl", "rb") as f:
    model = pickle.load(f)
# Crear el Enum de géneros
class Genre(Enum):
    Action = "Action"
    Adventure = "Adventure"
    Casual = "Casual"
    Early_Access = "Early Access"
    Free_to_Play = "Free to Play"
    Indie = "Indie"
    Massively_Multiplayer = "Massively Multiplayer"
    RPG = "RPG"
    Racing = "Racing"
    Simulation = "Simulation"
    Sports = "Sports"
    Strategy = "Strategy"
    Video_Production = "Video Production"
# Definir la ruta de predicción
@app.get("/predicción") 
def predict(metascore: float = None, earlyaccess: bool = None, Año: str = None, genero: Genre = None):
    # Validar que se hayan pasado los parámetros necesarios
    if metascore is None or Año is None or genero is None or earlyaccess is None:
        raise HTTPException(status_code=400, detail="Missing parameters")
    
    # Convertir el input en un DataFrame con las columnas necesarias para el modelo
    input_df = pd.DataFrame([[metascore, earlyaccess, Año, *[1 if genero.value == g else 0 for g in Genre._member_names_]]], columns=['metascore', 'year', 'early_access', *Genre._member_names_])
    
    # Verificar si el género es Free to Play
    if genero == Genre.Free_to_Play:
        # Devolver 0 como precio
        return {"price": 0, "RMSE del modelo": 10.00}
    else:
        # Realizar la predicción con el modelo
        try:
            price = model.predict(input_df)[0]
        except:
            raise HTTPException(status_code=400, detail="Invalid input")

        # Devolver el precio y el RMSE como salida
        return {"price": price, "RMSE del modelo": 10.00}
