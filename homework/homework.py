# flake8: noqa: E501
# Este proyecto implementa un modelo de red neuronal multicapa (MLP) para predecir
# el default (impago) de clientes de tarjetas de crédito basándose en sus características
# demográficas y historial de pagos.
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Descompone la matriz de entrada usando componentes principales.
#   El pca usa todas las componentes.
# - Escala la matriz de entrada al intervalo [0, 1].
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una red neuronal tipo MLP.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}

# # 1. Importación de librerías

# %%
import os
import json
import gzip
import pickle
import zipfile

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# %% [markdown]
# # 2. Función de limpieza de datos

# %%
def limpiar_datos(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Limpieza y preprocesamiento:
    - Elimina ID
    - Renombra columna objetivo a 'default'
    - Elimina NaN
    - Filtra EDUCATION != 0 y MARRIAGE != 0
    - EDUCATION > 4 -> 4 (others)
    """
    datos = datos.copy()
    datos = datos.drop("ID", axis=1)
    datos = datos.rename(columns={"default payment next month": "default"})
    datos = datos.dropna()
    datos = datos[(datos["EDUCATION"] != 0) & (datos["MARRIAGE"] != 0)]
    datos.loc[datos["EDUCATION"] > 4, "EDUCATION"] = 4
    return datos


# %% [markdown]
# # 3. Construcción del modelo (pipeline MLP)

# %%
def modelo() -> Pipeline:
    """
    Construye el pipeline:
    - OneHotEncoder para variables categóricas
    - StandardScaler para numéricas
    - SelectKBest
    - PCA
    - MLPClassifier
    (mismo orden y nombres de pasos que en tu código original)
    """
    categoricas = ["SEX", "EDUCATION", "MARRIAGE"]
    numericas = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    preprocesador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categoricas),
            ("scaler", StandardScaler(), numericas),
        ],
        remainder="passthrough",
    )

    seleccionar_k_mejores = SelectKBest(score_func=f_classif)

    pipeline = Pipeline(
        steps=[
            ("preprocesador", preprocesador),
            ("seleccionar_k_mejores", seleccionar_k_mejores),
            ("pca", PCA()),
            ("clasificador", MLPClassifier(max_iter=15000, random_state=42)),
        ]
    )

    return pipeline


def hiperparametros(
    modelo: Pipeline,
    n_splits: int,
    x_entrenamiento: pd.DataFrame,
    y_entrenamiento: pd.Series,
    puntuacion: str,
) -> GridSearchCV:
    """
    Optimiza los hiperparámetros usando GridSearchCV.
    (Misma grilla que en tu código original)
    """
    estimador = GridSearchCV(
        estimator=modelo,
        param_grid={
            "pca__n_components": [None],
            "seleccionar_k_mejores__k": [20],
            "clasificador__hidden_layer_sizes": [(50, 30, 40, 60)],
            "clasificador__alpha": [0.28],
            "clasificador__learning_rate_init": [0.001],
        },
        cv=n_splits,
        refit=True,
        scoring=puntuacion,
    )

    estimador.fit(x_entrenamiento, y_entrenamiento)
    return estimador


# %% [markdown]
# # 4. Métricas y matrices de confusión

# %%
def metricas(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    """
    Calcula métricas de entrenamiento y prueba:
    precision, balanced_accuracy, recall, f1_score.
    """
    y_entrenamiento_pred = modelo.predict(x_entrenamiento)
    y_prueba_pred = modelo.predict(x_prueba)

    metricas_entrenamiento = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(
            y_entrenamiento, y_entrenamiento_pred, average="binary"
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_entrenamiento, y_entrenamiento_pred
        ),
        "recall": recall_score(y_entrenamiento, y_entrenamiento_pred, average="binary"),
        "f1_score": f1_score(
            y_entrenamiento, y_entrenamiento_pred, average="binary"
        ),
    }

    metricas_prueba = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_prueba, y_prueba_pred, average="binary"),
        "balanced_accuracy": balanced_accuracy_score(y_prueba, y_prueba_pred),
        "recall": recall_score(y_prueba, y_prueba_pred, average="binary"),
        "f1_score": f1_score(y_prueba, y_prueba_pred, average="binary"),
    }

    return metricas_entrenamiento, metricas_prueba


def matriz_confusion(modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba):
    """
    Calcula matrices de confusión para train y test en el formato requerido.
    """
    y_entrenamiento_pred = modelo.predict(x_entrenamiento)
    y_prueba_pred = modelo.predict(x_prueba)

    cm_entrenamiento = confusion_matrix(y_entrenamiento, y_entrenamiento_pred)
    tn_ent, fp_ent, fn_ent, tp_ent = cm_entrenamiento.ravel()

    cm_prueba = confusion_matrix(y_prueba, y_prueba_pred)
    tn_test, fp_test, fn_test, tp_test = cm_prueba.ravel()

    matriz_entrenamiento = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(tn_ent),
            "predicted_1": int(fp_ent),
        },
        "true_1": {
            "predicted_0": int(fn_ent),
            "predicted_1": int(tp_ent),
        },
    }

    matriz_prueba = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(tn_test),
            "predicted_1": int(fp_test),
        },
        "true_1": {
            "predicted_0": int(fn_test),
            "predicted_1": int(tp_test),
        },
    }

    return matriz_entrenamiento, matriz_prueba


# %% [markdown]
# # 5. Guardado de modelo y métricas

# %%
def guardar_modelo(modelo) -> None:
    """
    Guarda el modelo entrenado en files/models/model.pkl.gz
    usando pickle + gzip.
    """
    os.makedirs("files/models", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", "wb") as f:
        pickle.dump(modelo, f)


def guardar_metricas(metricas_list) -> None:
    """
    Guarda una lista de diccionarios (métricas y matrices)
    en files/output/metrics.json, una por línea.
    """
    os.makedirs("files/output", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metrica in metricas_list:
            f.write(json.dumps(metrica) + "\n")


# %% [markdown]
# # 6. Carga de datos desde ZIP

# %%
def cargar_desde_zip(path_zip: str, internal_name: str) -> pd.DataFrame:
    """
    Carga un CSV específico desde un archivo .zip, igual que en tu código original.
    """
    with zipfile.ZipFile(path_zip, "r") as z:
        with z.open(internal_name) as f:
            df = pd.read_csv(f)
    return df


# Cargar conjuntos de prueba y entrenamiento (misma lógica que tenías)
df_prueba = cargar_desde_zip(
    "files/input/test_data.csv.zip", "test_default_of_credit_card_clients.csv"
)
df_entrenamiento = cargar_desde_zip(
    "files/input/train_data.csv.zip", "train_default_of_credit_card_clients.csv"
)


# %% [markdown]
# # 7. Ejecución principal del pipeline

# %%
if __name__ == "__main__":
    print("Iniciando limpieza de datos...")
    df_prueba = limpiar_datos(df_prueba)
    df_entrenamiento = limpiar_datos(df_entrenamiento)

    print("Dividiendo características y variable objetivo...")
    x_entrenamiento, y_entrenamiento = (
        df_entrenamiento.drop("default", axis=1),
        df_entrenamiento["default"],
    )
    x_prueba, y_prueba = df_prueba.drop("default", axis=1), df_prueba["default"]

    print("Construyendo pipeline del modelo...")
    pipeline_modelo = modelo()

    print("Optimizando hiperparámetros...")
    pipeline_modelo = hiperparametros(
        pipeline_modelo,
        n_splits=10,
        x_entrenamiento=x_entrenamiento,
        y_entrenamiento=y_entrenamiento,
        puntuacion="balanced_accuracy",
    )

    print("Guardando modelo entrenado...")
    guardar_modelo(pipeline_modelo)

    print("Calculando métricas de evaluación...")
    metricas_entrenamiento, metricas_prueba = metricas(
        pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba
    )

    print("Calculando matrices de confusión...")
    matriz_entrenamiento, matriz_prueba = matriz_confusion(
        pipeline_modelo, x_entrenamiento, y_entrenamiento, x_prueba, y_prueba
    )

    print("Guardando métricas y matrices...")
    guardar_metricas(
        [
            metricas_entrenamiento,
            metricas_prueba,
            matriz_entrenamiento,
            matriz_prueba,
        ]
    )

    print("¡Proceso completado exitosamente!")
    print("- Modelo guardado en: files/models/model.pkl.gz")
    print("- Métricas guardadas en: files/output/metrics.json")
