import pandas as pd
import numpy as np


def construir_features_vibracion(
    df,
    motor_col,
    fecha_col,
    vibracion_col
):

    # Copia del DataFrame
    df = df.copy()

    # Convertir fecha a datetime
    df[fecha_col] = pd.to_datetime(df[fecha_col])

    # Ordenar por motor y fecha
    df = df.sort_values(
        [motor_col, fecha_col]
    ).reset_index(drop=True)

    # ====================================
    # Features temporales
    # ====================================

    df['lag_1'] = (
        df.groupby(motor_col)[vibracion_col]
        .shift(1)
    )

    df['lag_7'] = (
        df.groupby(motor_col)[vibracion_col]
        .shift(7)
    )

    df['tendencia_3d'] = (
        df[vibracion_col]
        - df.groupby(motor_col)[vibracion_col]
        .shift(3)
    )

    # ====================================
    # Eliminar NaN
    # ====================================

    df = df.dropna(
        subset=[
            'lag_1',
            'lag_7',
            'tendencia_3d'
        ]
    )

    # Reiniciar índice
    df = df.reset_index(drop=True)

    return df
