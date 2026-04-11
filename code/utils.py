import plotnine as p9
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

def plot_contingency_table(labels1, labels2, method1_name, method2_name):
    # Preparamos los datos para la tabla de contingencia
    # Creamos un DataFrame con las etiquetas de ambos métodos para cada nodo
    df_compare = pd.DataFrame({
        method1_name: labels1,
        method2_name: labels2
    })

    # Calculamos la tabla de contingencia entre las etiquetas de los dos métodos
    contingency_df = pd.crosstab(df_compare[method1_name], df_compare[method2_name]).reset_index()
    plot_data = contingency_df.melt(id_vars=method1_name, var_name=method2_name, value_name='Count')

    # Convertimos las etiquetas a string para una mejor visualización
    plot_data[method1_name] = plot_data[method1_name].astype(str)
    plot_data[method2_name] = plot_data[method2_name].astype(str)

    # Creamos la tabla de contingencia como un gráfico de calor
    contingency_plot = (
        p9.ggplot(plot_data, p9.aes(x=method2_name, y=method1_name, fill='Count'))
        + p9.geom_tile()
        + p9.geom_text(p9.aes(label='Count'), size=8) # Muestra el número de nodos en cada celda
        + p9.scale_fill_cmap(cmap_name='YlOrRd')
        + p9.theme_minimal()
        + p9.theme(
            panel_grid=p9.element_blank(),        # Removes all grid lines
            panel_grid_major=p9.element_blank(),  # Ensures major grid is gone
            panel_grid_minor=p9.element_blank(),  # Ensures minor grid is gone
        )
        + p9.labs(
            title='Tabla de Contingencia: Solapamiento de Comunidades',
            subtitle=f'Comparación entre {method1_name} y {method2_name}',
            x=f'Comunidades {method2_name}',
            y=f'Comunidades {method1_name}',
            fill='Nodos'
        )
    )

    return contingency_plot

def contagion_cascade(g_infected, beta):
    # Preparamos los datos para la visualización de cascada de contagio
    # Mapeamos cada nodo a su periodo {node_id: period_id}
    node_to_period = {}
    for node in g_infected.nodes(data=True):
        node_to_period[node[0]] = node[1]['t']

    # Obtenemos la lista de nodos en el mismo orden que g_infected
    # y sus respectivas etiquetas de periodo de infección
    nodes = list(g_infected.nodes())
    periods = [node_to_period[node] for node in nodes]
    num_periods = len(set(periods))

    # Elegimos la paleta de colores
    cmap = plt.colormaps['tab20']

    period_series = pd.Categorical(periods)
    colors = [cmap(period_series.codes[i]) for i in range(len(periods))]

    # Calculamos el layout del grafo
    pos = nx.spring_layout(g_infected, k=0.15, seed=42, iterations=50) 

    # Configuramos el tamaño de la figura para la visualización
    plt.figure(figsize=(14, 12))
    ax = plt.gca()

    # Graficamos los nodos con colores según su comunidad
    nx.draw_networkx_nodes(g_infected, pos, 
                        nodelist=nodes,
                        node_size=30, # Tamaño del nodo
                        node_color=colors, # Color del nodo según su periodo
                        alpha=0.8
                        )

    # Graficamos los enlaces con alta transparencia para evitar el ruido visual
    nx.draw_networkx_edges(g_infected, pos, alpha=1, edge_color='gray')

    # Titulo
    plt.title(f'Cascada de contagios, Beta = {beta}\n'
            f'Nodos: {g_infected.number_of_nodes()}, Periodos: {num_periods},', 
            fontsize=16)
    plt.axis('off') # Eliminamos los ejes para una visualización más limpia

    # Agregamos leyenda
    for i in range(num_periods):
        ax.scatter([], [], c=[cmap(i)], label=f'Periodo {i}')

    plt.legend(loc='upper right', title="Periodos", fontsize=10)

    plt.tight_layout()
    plt.show()

