import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_resumen_estudiantes():
    
    n = random.randint(10, 30)
    
    grupos = ['A', 'B', 'C']
    
    df = pd.DataFrame({
        'grupo': np.random.choice(grupos, n),
        'nota': np.random.uniform(0, 5, n)
    })
    
    input_data = {'df': df.copy()}
    
    output_data = df.groupby('grupo')['nota'].mean().reset_index(name='promedio')
    
    return input_data, output_data
