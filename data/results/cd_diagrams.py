import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from autorank import autorank
import warnings
import matplotlib.lines as mlines

# Ignorar advertencias de Friedman cuando los datos tienen empates
warnings.filterwarnings("ignore")

# DICCIONARIO FIJO DE ESTILOS: Color (Familia) + Linestyle/Marker (Arquitectura)
STYLE_MAP = {
    'DistilBERT':                 {'color': '#1f77b4', 'ls': '-',  'marker': 'o'},
    'SparseDistilBERT':           {'color': '#1f77b4', 'ls': '--', 'marker': 's'},
    'BiLBERT-Distil (Completo)':  {'color': '#1f77b4', 'ls': ':',  'marker': 'X'},
    'BiLBERT-Distil (Congelado)': {'color': '#1f77b4', 'ls': '-.', 'marker': '^'},
    'BERTLarge':                  {'color': '#d62728', 'ls': '-',  'marker': 'o'},
    'SparseBERTLarge':            {'color': '#d62728', 'ls': '--', 'marker': 's'},
    'BiLBERT-Large (Completo)':   {'color': '#d62728', 'ls': ':',  'marker': 'X'},
    'BiLBERT-Large (Congelado)':  {'color': '#d62728', 'ls': '-.', 'marker': '^'},
}

def draw_beautiful_cd_diagram_on_ax(ranks, cd, ax, title):
    # Ordenados de mejor a peor (de izquierda a derecha en el eje)
    ranks = ranks.sort_values()
    models = ranks.index.tolist()
    n_models = len(models)
    
    actual_min = ranks.min()
    actual_max = ranks.max()
    drange = actual_max - actual_min
    if drange == 0: drange = 1.0
    
    # Zoom en las marcas del eje (0.1, 0.2, etc.)
    if drange <= 1.0:
        step = 0.1
    elif drange <= 2.0:
        step = 0.2
    elif drange <= 5.0:
        step = 0.5
    else:
        step = 1.0
        
    low_tick = np.floor(actual_min / step) * step
    high_tick = np.ceil(actual_max / step) * step
    
    axis_start = low_tick - (step * 0.1)
    axis_end = high_tick + (step * 0.1)
    
    # Dibujar eje principal
    ax.plot([axis_start, axis_end], [0, 0], color='k', lw=1.5)
    
    for tick in np.arange(low_tick, high_tick + (step/2), step):
        ax.plot([tick, tick], [0, 0.05], color='k', lw=1)
        format_str = "{:.1f}" if step < 1.0 else "{:.0f}"
        ax.text(tick, 0.08, format_str.format(tick), ha='center', va='bottom', fontsize=14)
        
    cliques = []
    for i in range(n_models):
        for j in range(n_models-1, i, -1):
            if ranks.iloc[j] - ranks.iloc[i] <= cd:
                cliques.append((i, j))
                break
                
    max_cliques = []
    for c in cliques:
        is_subset = False
        for oc in cliques:
            if c != oc and c[0] >= oc[0] and c[1] <= oc[1]:
                is_subset = True
                break
        if not is_subset:
            max_cliques.append(c)
            
    # Líneas negras
    clique_y_start = -0.15
    clique_step = 0.1
    for idx, (start_idx, end_idx) in enumerate(max_cliques):
        start_rank = ranks.iloc[start_idx]
        end_rank = ranks.iloc[end_idx]
        y_clique = clique_y_start - (idx * clique_step)
        
        ax.plot([start_rank, end_rank], [y_clique, y_clique], color='black', lw=4, zorder=4)
        ax.plot([start_rank, start_rank], [y_clique - 0.03, y_clique + 0.03], color='black', lw=1.5)
        ax.plot([end_rank, end_rank], [y_clique - 0.03, y_clique + 0.03], color='black', lw=1.5)
        
    min_clique_y = clique_y_start - (len(max_cliques) * clique_step)
    
    # ========================================================
    # TU SOLUCIÓN: CASCADA PURA HACIA LA IZQUIERDA
    # ========================================================
    line_y_start = min_clique_y - 0.25
    y_step = 0.35 
    
    # Todos los modelos irán a esta coordenada X (a la izquierda del eje)
    text_margin = max(drange * 0.1, 0.2)
    x_target = axis_start - text_margin
    
    min_y_pos = 0
    
    for i, (model, rank) in enumerate(ranks.items()):
        style = STYLE_MAP.get(model, {'color': 'black', 'ls': '-', 'marker': 'o'})
        c = style['color']
        ls = style['ls']
        m = style['marker']
        
        ax.scatter(rank, 0, color=c, marker=m, zorder=6, s=100)
        
        y_drop = line_y_start - (i * y_step)
        min_y_pos = min(min_y_pos, y_drop)
        
        # Caída vertical
        ax.plot([rank, rank], [0, y_drop], color=c, lw=2.5, linestyle=ls)
        
        # Línea horizontal SIEMPRE hacia la izquierda (x_target)
        ax.plot([rank, x_target], [y_drop, y_drop], color=c, lw=2.5, linestyle=ls)
        
        # Texto a la izquierda de la línea, alineado a la derecha (ha='right')
        offset = max(drange * 0.02, 0.05)
        ax.text(x_target - offset, y_drop, f"{model} ({rank:.2f})",
                ha='right', va='center', color=c, fontsize=13, fontweight='bold', # FUENTE REDUCIDA A 13
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=3), zorder=5)
                
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.axis('off')
    
    # LÍMITES DE LA CÁMARA (ZOOM)
    # Reducimos el margen izquierdo para que el texto ocupe ~30% y el eje el ~70%
    margin_text = max(drange * 0.3, 1.2) 
    ax.set_xlim(axis_start - margin_text, axis_end + max(drange * 0.05, 0.1))
    
    top_y = 0.25
    ax.set_ylim(min_y_pos - 0.2, top_y)

def main():
    print("Iniciando generación de CD-Diagrams (2x2) ULTRAWIDE con estilos fijos...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    hist_file = os.path.join(script_dir, 'tfg_results.xlsx')
    new_file = os.path.join(script_dir, 'tfg_results_mymodels.xlsx')
    cong_file = os.path.join(script_dir, 'tfg_results_mymodels_congelado.xlsx')
    
    output_dir = os.path.join(script_dir, 'exported_results_mymodels')
    cd_dir = os.path.join(output_dir, 'cd_diagrams')
    os.makedirs(cd_dir, exist_ok=True)
    
    if not os.path.exists(hist_file) or not os.path.exists(new_file) or not os.path.exists(cong_file):
        print("Error: No se encontraron los archivos Excel necesarios.")
        return

    df_hist = pd.read_excel(hist_file)
    df_new = pd.read_excel(new_file)
    df_cong = pd.read_excel(cong_file)
    
    df_hist['Model'] = df_hist['Model'].replace({'Transformer DistilBERT': 'DistilBERT', 'Transformer BERT': 'BERTLarge'})
    df_new['Model'] = df_new['Model'].replace({
        'Transformer SparseDistilBERT': 'SparseDistilBERT', 'Transformer SparseBERTLarge': 'SparseBERTLarge',
        'Transformer BiLBERTDistil': 'BiLBERT-Distil (Completo)', 'Transformer BiLBERTLarge': 'BiLBERT-Large (Completo)'
    })
    df_cong['Model'] = df_cong['Model'].replace({
        'Transformer BiLBERTDistil': 'BiLBERT-Distil (Congelado)', 'Transformer BiLBERTLarge': 'BiLBERT-Large (Congelado)'
    })
    
    model_list = [
        'DistilBERT', 'SparseDistilBERT', 'BiLBERT-Distil (Completo)', 'BiLBERT-Distil (Congelado)',
        'BERTLarge', 'SparseBERTLarge', 'BiLBERT-Large (Completo)', 'BiLBERT-Large (Congelado)'
    ]
    
    df_all = pd.concat([
        df_hist[df_hist['Model'].isin(model_list)],
        df_new[df_new['Model'].isin(model_list)],
        df_cong[df_cong['Model'].isin(['BiLBERT-Distil (Congelado)', 'BiLBERT-Large (Congelado)'])]
    ], ignore_index=True)
    
    metrics = ['ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore', 'ExecTime']
    for m in metrics:
        if df_all[m].dtype == object:
            df_all[m] = df_all[m].astype(str).str.replace(',', '.')
        df_all[m] = pd.to_numeric(df_all[m], errors='coerce')
    
    datasets = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    letters = ['a)', 'b)', 'c)', 'd)']
    
    for metric in metrics:
        print(f"Procesando métrica: {metric}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(22, 14))
        axes = axes.flatten()
        
        sort_order = 'ascending' if metric == 'ExecTime' else 'descending'
        metric_name = "Tiempo de Ejecución" if metric == 'ExecTime' else metric
        
        fig.suptitle(f'Diagrama de Diferencias Críticas - {metric_name}', fontsize=24, fontweight='bold', y=0.98)
        
        for i, dataset in enumerate(datasets):
            ax = axes[i]
            title = f"{letters[i]} {dataset}"
            
            df_sub = df_all[df_all['Dataset'] == dataset].copy()
            
            if metric in ['ROUGE_L', 'BERTScore']:
                df_sub = df_sub[df_sub['Status'] == 'TP']
                title += " (Solo TP)"
                
            df_pivot = df_sub.pivot(index='Question ID', columns='Model', values=metric).dropna()
            
            if len(df_pivot) < 10:
                ax.axis('off')
                ax.set_title(f"{title}\n(Datos insuficientes)", fontsize=14)
                continue
                
            try:
                result = autorank(df_pivot, alpha=0.05, verbose=False, order=sort_order)
                draw_beautiful_cd_diagram_on_ax(result.rankdf['meanrank'], result.cd, ax, title)
            except Exception as e:
                ax.axis('off')
                ax.set_title(f"{title}\n(Error al converger)", fontsize=14)
                
        # Legend
        legend_handles = []
        for model_name in model_list:
            style = STYLE_MAP.get(model_name, {'color': 'black', 'ls': '-', 'marker': 'o'})
            # Creamos un elemento visual de "línea" para representarlo en la leyenda
            handle = mlines.Line2D([], [], color=style['color'], linestyle=style['ls'], 
                                   marker=style['marker'], markersize=10, linewidth=2.5, 
                                   label=model_name)
            legend_handles.append(handle)
            
        # Añadimos la leyenda al Figure global ( ncol=4 para 2 filas x 4 columnas fijos)
        fig.legend(handles=legend_handles, loc='lower center', ncol=4, fontsize=14, 
                   bbox_to_anchor=(0.5, 0.01), frameon=True, 
                   title="Codificación de Modelos fijos (Color = Familia Base | Trazo/Marcador = Arquitectura)", title_fontsize=16)

       # 1. Ajustamos rect para dejar el 12% inferior de la pantalla para la leyenda
        plt.tight_layout(rect=[0, 0.12, 1, 0.95])
        
        # 2. SEPARACIÓN LATERAL DINÁMICA
        # Si es la gráfica de Tiempos (rango numérico ancho), separamos las columnas. Si no, las pegamos.
        current_wspace = 0.35 if metric == 'ExecTime' else 0.1
        plt.subplots_adjust(wspace=current_wspace, hspace=0.3)
        
        filename = f"cd_diagram_{metric}.png"
        plt.savefig(os.path.join(cd_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"\n¡Éxito! Todas las matrices 2x2 de CD-Diagrams se han guardado en '{cd_dir}'.")

if __name__ == "__main__":
    main()