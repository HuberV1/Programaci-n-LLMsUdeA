import pandas as pd
import numpy as np
import random
from sklearn.linear_model import LogisticRegression

def generar_caso_de_uso_entrenar_clasificador():
    
    n_rows = random.randint(20, 50)
    n_features = random.randint(2, 5)
    
    X = np.random.randn(n_rows, n_features)
    y = np.random.randint(0, 2, n_rows)
    
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
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    return input_data, model
