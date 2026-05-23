import pandas as pd
import numpy as np

def transformar_tempo_ciclico(df, col_hora):

    df = df.copy()

    angulo = 2 * np.pi * df[col_hora] / 24

    df['hora_sin'] = np.sin(angulo)
    df['hora_cos'] = np.cos(angulo)

    df = df.drop(columns=[col_hora])

    return df
