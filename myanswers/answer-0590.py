import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

def incertidumbre_prediccion_gaussiana(X_train, y_train, X_test):

    # Crear modelo
    gpr = GaussianProcessRegressor(random_state=42)

    # Entrenar modelo
    gpr.fit(X_train, y_train)

    # Obtener predicciones y desviación estándar
    _, std = gpr.predict(X_test, return_std=True)

    # Retornar únicamente la incertidumbre
    return std
