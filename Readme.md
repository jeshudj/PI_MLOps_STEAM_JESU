<p align=center><img src="src\001logohenry.png"><p>

# <h1 align=center> **APLICACION DE PREDICCION Y CONSULTAS DE JUEGOS STEAM** </h1>
# <h1 align=center> **PROYECTO INDIVIDUAL Nº1** </h1>
# <h1 align=center> **Jesus Marceliano Neira** </h1>

# <h1 align=center>**`Machine Learning Operations (MLOps)`**</h1>

<p align="center">
  <img src="src\002MLOpslogo.png" alt="Diagrama de Flujo">
</p>



## ¡Bienvenido a mi aplicación de prediccion y consultas de juegos STEAM! Aquí, podrás explorar y descubrir juegos de acuerdo a tus gustos. La aplicacion de prediccion utilizará técnicas de ***Machine Learning*** para brindarte sugerencias, basadas en un historial de Juesgo de STEAM. 

<hr>  

## **Descripción del proyecto**

## Contexto

Como Data Scientist Nuestro objetivo es desarrollar una aplicación de prediccion y consultas que permita a los usuarios descubrir nuevos contenidos relevantes y disfrutar de una experiencia de busqueda personalizada por años.



<p align="center">
  <img src="src\003transformer.png" alt="Diagrama de tranformacion">
</p> 


## Transformaciones ETL

En el archivo ETL_STEAM.ipynb que se proporciona hay una serie de pasos que se realizó para extraer, transformar y cargar datos en un DataFrame llamado 'clean_games'.



## Análisis exploratorio de los datos EDA

Ya los datos están limpios, ahora es tiempo de investigar las relaciones que hay entre las variables de los datasets, ver si hay outliers o anomalías (que no tienen que ser errores necesariamente), y ver si hay algún patrón interesante que valga la pena explorar en un análisis posterior. Las nubes de palabras dan una buena idea de cuáles palabras son más frecuentes en los títulos, se deja capturas obtenidas del DataFrame llamado 'clean_games' y se puede explorar en el archivo llamado 'EDA_STEAM.ipynb'.


<p align="center">
  <img src="src\004Nubepalabras.png" >
</p>

<p align="center">
  <img src="src\005Histogram.png" >
</p>



## API en desarrollo: 6 funciones API con FastAPI

Se Propone disponibilizar los datos usando el framework ***FastAPI***. Las consultas que se propones son las siguientes:

Deben crear 6 funciones para los endpoints que se consumirán en la API, recuerden que deben tener un decorador por cada una (@app.get(‘/’)).
  
def genero( Año: str ): Se ingresa un año y devuelve una lista con los 5 géneros más ofrecidos en el orden correspondiente.

def juegos( Año: str ): Se ingresa un año y devuelve una lista con los juegos lanzados en el año.

def specs( Año: str ): Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.

def earlyacces( Año: str ): Cantidad de juegos lanzados en un año con early access.

def sentiment( Año: str ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.

                    Ejemplo de retorno: {Mixed = 182, Very Positive = 120, Positive = 278}

def metascore( Año: str ): Top 5 juegos según año con mayor metascore.




**Modelo de predicción:**

Una vez que toda la data es consumible por la API, está lista para consumir por los departamentos de Analytics y Machine Learning, y nuestro EDA nos permite entender bien los datos a los que tenemos acceso, es hora de entrenar nuestro modelo de machine learning para armar un modelo de predicción. El mismo deberá basarse en características como Género, Año, Metascore y/o las que creas adecuadas. Tu líder pide que el modelo derive en un GET/POST en la API simil al siguiente formato:

def predicción( genero, earlyaccess = True/False, (Variables que elijas) ): Ingresando estos parámetros, deberíamos recibir el precio y RMSE.

**Este repositorio incluye:**

+ Cuadernos de Jupyter para su visualización<br/>
+ Un proceso ETL paso a paso<br/>
+ Un análisis exploratorio de datos (EDA)<br/>
+ Desarrollo de una API<br/>
+ Implementación<br/>



## Detalles adicionales del proyecto

Aquí encontrarás información adicional y recursos relacionados de mi proyecto:

1. `Video explicativo:` Se ha creado un [video explicativo](https://www.youtube.com/watch?v=69yCXB35Msk&t=1s)  donde te muestro algunas funciones de mi proyecto con el uso de la API.

2. `Acceso a la API:` En el Siguiente [enlace de la API](https://pi-mlops-steam-jesu.onrender.com/docs) podras encontrar las funciones de este proyecto.

3. `Obtención de datos originales:` Si te interesa en obtener acceso a los datos originales utilizados en este proyecto de análisis, puedes ir al siguiente [enlace de descarga](https://drive.google.com/drive/folders/1HqBG2-sUkz_R3h1dZU5F2uAzpRn7BSpj) para que puedas explorar y analizar los datos por tu cuenta.

4. `Acceso rápido:`
- Visualize EDA [EDA_STEAM](EDA_STEAM.ipynb) notebook.
- Visualize ETL [ETL_STEAM](ETL_STEAM.ipynb) notebook.
- Visualize API  [`MAIN.PY`](main.py)

<br/>

