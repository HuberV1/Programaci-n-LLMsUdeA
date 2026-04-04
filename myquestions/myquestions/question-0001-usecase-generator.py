mport pandas as pd
import numpy as np
import random

def generar_caso_de_uso_limpiar_ventas():
    
    n_rows = random.randint(8, 20)
    n_cols = random.randint(3, 6)
    
    data = np.random.randn(n_rows, n_cols)
    cols = [f'col_{i}' for i in range(n_cols)]
    
    df = pd.DataFrame(data, columns=cols)
    
    for col in df.columns:
        if random.random() < 0.5:
            df.loc[df.sample(frac=0.6).index, col] = np.nan
    
    df = pd.concat([df, df.iloc[:2]], ignore_index=True)
    
    input_data = {'df': df.copy()}
    
    df_clean = df.drop_duplicates()
    cols_to_keep = df_clean.columns[df_clean.isnull().mean() <= 0.5]
    df_clean = df_clean[cols_to_keep]
    
    return input_data, df_clean
