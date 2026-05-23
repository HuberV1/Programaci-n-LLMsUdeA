import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def segmentar_comportamiento_robusto(df, n_clusters=3):

    # Copia para limpieza
    df_limpio = df.copy()

    # ====================================
    # Reemplazar infinitos por el máximo
    # de cada columna
    # ====================================

    for col in df_limpio.columns:

        maximo_columna = (
            df_limpio.loc[
                ~np.isinf(df_limpio[col]),
                col
            ].max()
        )

        df_limpio[col] = np.where(
            np.isinf(df_limpio[col]),
            maximo_columna,
            df_limpio[col]
        )

    # ====================================
    # Completar NaN con la mediana
    # ====================================

    for col in df_limpio.columns:

        mediana = df_limpio[col].median()

        df_limpio[col] = (
            df_limpio[col]
            .fillna(mediana)
        )

    # ====================================
    # Reducción dimensional con t-SNE
    # ====================================

    tsne = TSNE(
        n_components=2,
        random_state=42
    )

    X_tsne = tsne.fit_transform(df_limpio)

    # ====================================
    # Clustering jerárquico
    # ====================================

    clustering = AgglomerativeClustering(
        n_clusters=n_clusters
    )

    labels = clustering.fit_predict(X_tsne)

    # ====================================
    # Calcular silhouette score
    # ====================================

    score = silhouette_score(
        X_tsne,
        labels
    )

    score = round(score, 4)

    # ====================================
    # Resultado final
    # ====================================

    resultado = df.copy()

    resultado["subsegmento_id"] = labels

    return resultado, score
