import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def generar_caso_de_uso_evaluar_modelo():
    
    n_rows = random.randint(20, 50)
    n_features = random.randint(2, 5)
    
    X = np.random.randn(n_rows, n_features)
    coef = np.random.randn(n_features)
    
    y = X @ coef + np.random.randn(n_rows) * 0.1
    
    cols = [f'f{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    
    target_col = 'target'
    df[target_col] = y
    
    input_data = {
        'df': df.copy(),
        'target_col': target_col
    }
    
    X_train = df.drop(columns=[target_col])
    y_train = df[target_col]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_train)
    mse = mean_squared_error(y_train, y_pred)
    
    return input_data, mse
