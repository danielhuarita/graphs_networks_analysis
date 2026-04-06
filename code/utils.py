import plotnine as p9
import pandas as pd
import numpy as np

# Plot corr matrix
def corr_plot(data, numeric_var,fig_size,corr_filter=0):
    # Calcular matriz de correlación
    corr_matrix = data[numeric_var].corr()
    col_order = corr_matrix.columns.tolist()

    # Creamos una matriz 'mask' para poder quedarnos solo con el triangulo inferior
    mask = np.zeros_like(corr_matrix, dtype=bool)
    # Nos quedamos con el triangulo inferior y la diagonal
    mask[np.triu_indices_from(mask, k=0)] = True

    # Utilizamos mask para filtrar y formateamos la matriz para plotnine
    corr_matrix = corr_matrix.mask(mask).stack().reset_index(name='value')
    corr_matrix.columns = ['var1', 'var2', 'value']

    corr_matrix['var1'] = pd.Categorical(corr_matrix['var1'], categories=col_order)
    corr_matrix['var2'] = pd.Categorical(corr_matrix['var2'], categories=col_order)
    corr_matrix = corr_matrix[np.abs(corr_matrix['value'])>=corr_filter]

    corr_plot = (
        p9.ggplot(corr_matrix, p9.aes(x='var1', y='var2', fill='value'))
        + p9.geom_tile()  # This creates the squares
        + p9.geom_text(p9.aes(label='value.round(2)'), size=8) # Add coefficients
        + p9.scale_fill_gradient2(
            low='#d7191c', 
            mid='#ffffbf', 
            high="#05b402", 
            midpoint=0, 
            limits=[-1, 1]
        )
        + p9.theme_minimal()
        + p9.theme(
            axis_text_x=p9.element_text(rotation=45, hjust=1),
            axis_title=p9.element_blank(),
            figure_size=fig_size
        )
        + p9.labs(title="Correlation Matrix", fill="Corr")
    )

    return corr_plot