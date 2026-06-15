import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections

MODEL_ORDER = [
    'BiLBERT-Distil (Congelado)', 
    'BiLBERT-Distil (Completo)', 
    'BiLBERT-Large (Congelado)', 
    'BiLBERT-Large (Completo)'
]

# Colors
COLORS = {
    'BiLBERT-Distil (Congelado)': '#aec7e8', 
    'BiLBERT-Distil (Completo)': '#1f77b4',  
    'BiLBERT-Large (Congelado)': '#ff9896',  
    'BiLBERT-Large (Completo)': '#d62728'   
}

HATCHES = {
    'BiLBERT-Distil (Congelado)': 'xx',
    'BiLBERT-Distil (Completo)': 'xx',
    'BiLBERT-Large (Congelado)': 'xx',
    'BiLBERT-Large (Completo)': 'xx'
}

LINE_STYLES = {
    'BiLBERT-Distil (Congelado)': '--',
    'BiLBERT-Distil (Completo)': '-',
    'BiLBERT-Large (Congelado)': '--',
    'BiLBERT-Large (Completo)': '-'
}

MARKERS = {
    'BiLBERT-Distil (Congelado)': 'X',
    'BiLBERT-Distil (Completo)': 'X',
    'BiLBERT-Large (Congelado)': 'X',
    'BiLBERT-Large (Completo)': 'X'
}

def apply_hatches_to_axes(ax, plot_data, plot_type='box'):
    num_datasets = len(plot_data['Dataset'].unique())
    models = plot_data['Model'].cat.categories
    models_repeated = list(models) * num_datasets
    
    if plot_type == 'box':
        boxes = [p for p in ax.patches if isinstance(p, mpatches.PathPatch)]
        if boxes:
            boxes.sort(key=lambda p: p.get_extents().get_points()[:,0].mean())
            for box, model in zip(boxes, models_repeated):
                hatch = HATCHES.get(model, '')
                if hatch:
                    box.set_hatch(hatch)
                        
    elif plot_type == 'violin':
        poly_cols = [c for c in ax.collections if isinstance(c, mcollections.PolyCollection)]
        if poly_cols:
            poly_cols.sort(key=lambda c: c.get_paths()[0].get_extents().get_points()[:,0].mean())
            for col, model in zip(poly_cols, models_repeated):
                hatch = HATCHES.get(model, '')
                if hatch:
                    col.set_hatch(hatch)

def load_ablation_data(completo_filepath, congelado_filepath):
    print("Loading data")
    df_comp = pd.read_excel(completo_filepath)
    df_cong = pd.read_excel(congelado_filepath)
    
    rename_dict = {
        'Transformer BiLBERTDistil': 'BiLBERT-Distil',
        'Transformer BiLBERTLarge': 'BiLBERT-Large',
    }
    
    df_comp['Model'] = df_comp['Model'].replace(rename_dict)
    df_cong['Model'] = df_cong['Model'].replace(rename_dict)
    
    # Filter BiL models
    df_comp_bil = df_comp[df_comp['Model'].isin(['BiLBERT-Distil', 'BiLBERT-Large'])].copy()
    df_cong_bil = df_cong[df_cong['Model'].isin(['BiLBERT-Distil', 'BiLBERT-Large'])].copy()
    
    df_comp_bil['Model'] = df_comp_bil['Model'] + ' (Completo)'
    df_cong_bil['Model'] = df_cong_bil['Model'] + ' (Congelado)'
    
    return pd.concat([df_cong_bil, df_comp_bil], ignore_index=True)

def create_boxplots(df, output_dir):
    print("Generating Box Plots...")
    metrics = ['ExecTime', 'ROUGE_L', 'BERTScore']
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    
    box_dir = os.path.join(output_dir, 'box_plots')
    os.makedirs(box_dir, exist_ok=True)

    df['Dataset'] = pd.Categorical(df['Dataset'], categories=dataset_order, ordered=True)
    df['Model'] = pd.Categorical(df['Model'], categories=MODEL_ORDER, ordered=True)

    for metric in metrics:
        plt.figure(figsize=(14, 8))
        display_metric = 'Tiempo de Ejecución' if metric == 'ExecTime' else metric
        
        if metric in ['ROUGE_L', 'BERTScore']:
            plot_data = df[df['Status'] == 'TP']
            title_suffix = " (Solo Verdaderos Positivos)"
            ax = sns.violinplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette=COLORS, inner='quartile', density_norm='width', cut=0)
            apply_hatches_to_axes(ax, plot_data, plot_type='violin')
        else:
            plot_data = df
            title_suffix = ""
            ax = sns.boxplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette=COLORS, showfliers=True)
            apply_hatches_to_axes(ax, plot_data, plot_type='box')
        
        if metric == 'ExecTime':
            plt.yscale('log')
            plt.ylabel(f'{display_metric} (Escala Logarítmica)', fontsize=14)
        else:
            plt.ylabel(f'{display_metric}', fontsize=14)

        plt.title(f'Distribución Comparativa de {display_metric} (Estudio Ablación){title_suffix}', fontsize=16, fontweight='bold')
        plt.xlabel('Base de Datos', fontsize=14)
        plt.xticks(fontsize=12, rotation=0) 
        plt.yticks(fontsize=12)
        
        legend_elements = [mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES[m], label=m) for m in MODEL_ORDER]
        ax.legend(handles=legend_elements, title='Modelo', fontsize=11, title_fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
        
        plt.tight_layout()
        
        png_filename = f"ablation_boxplot_{metric}.png"
        plt.savefig(os.path.join(box_dir, png_filename), dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_tables(df, output_dir):
    print("Generating Summary Tables (Excel)...")
    metrics_to_mean = ['ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore']    
    datasets = df['Dataset'].unique()
    
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)
    excel_path = os.path.join(table_dir, 'ablation_summary_tables.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for dataset in datasets:
            df_subset = df[df['Dataset'] == dataset]
            
            all_means = df_subset.groupby('Model', observed=False)[metrics_to_mean].mean().reset_index()
            all_means.columns = ['Model'] + [f"{c} (Global)" for c in metrics_to_mean]
            
            df_tp = df_subset[df_subset['Status'] == 'TP']
            tp_means = df_tp.groupby('Model', observed=False)[metrics_to_mean].mean().reset_index()
            tp_means.columns = ['Model'] + [f"{c} (TP)" for c in metrics_to_mean]
            
            summary_df = pd.merge(all_means, tp_means, on='Model')

            summary_df = summary_df.round(5)
            summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=MODEL_ORDER, ordered=True)
            summary_df = summary_df.sort_values('Model').reset_index(drop=True)
            
            summary_df_es = summary_df.rename(columns={'Model': 'Modelo'})
            
            sheet_name = str(dataset)[:31]
            summary_df_es.to_excel(writer, sheet_name=sheet_name, index=False)

def create_time_summary_table(df, output_dir):
    print("Generating Time Summary Table (Excel)...")
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)
    excel_path = os.path.join(table_dir, 'ablation_time_summary.xlsx')
    
    time_table = df.pivot_table(index='Model', columns='Dataset', values='ExecTime', aggfunc='mean', observed=False)
    
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    time_table = time_table.reindex(MODEL_ORDER).reindex(columns=dataset_order).reset_index()
    time_table = time_table.round(5) 

    time_table.to_excel(excel_path, index=False, sheet_name='Tiempos_Medios')

def create_confusion_matrixes(df, output_dir):
    print("Generating Composite Confusion Matrix (Ablation 2x2)...")
    
    matrix_order = [
        ['BiLBERT-Distil (Congelado)', 'BiLBERT-Distil (Completo)'],
        ['BiLBERT-Large (Congelado)', 'BiLBERT-Large (Completo)']
    ]
    letters = [['a)', 'b)'], ['c)', 'd)']]
    
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    fig.suptitle('Estudio de Ablación: Matrices de Confusión', fontsize=16, fontweight='bold', y=1.05)
    
    for row in range(2):
        for col in range(2):
            model = matrix_order[row][col]
            letter = letters[row][col]
            ax = axes[row][col]
            
            df_subset = df[df['Model'] == model]
            status_counts = df_subset['Status'].value_counts().to_dict()
            
            tp = int(status_counts.get('TP', 0))
            tn = int(status_counts.get('TN', 0))
            fp = int(status_counts.get('FP', 0))
            fn = int(status_counts.get('FN', 0))
            
            total = tp + tn + fp + fn
            tp_pct = round((tp / total) * 100) if total > 0 else 0
            tn_pct = round((tn / total) * 100) if total > 0 else 0
            fp_pct = round((fp / total) * 100) if total > 0 else 0
            fn_pct = round((fn / total) * 100) if total > 0 else 0
            accuracy = tp_pct + tn_pct
            
            ax.set_xlim(0, 2)
            ax.set_ylim(0, 2)
            ax.set_aspect('equal') 
            
            ax.axhline(1, color='gray', linewidth=1.5)
            ax.axvline(1, color='gray', linewidth=1.5)
            
            ax.text(0.5, 1.5, f"TP\n{tp_pct}%", ha='center', va='center', color='green', fontsize=13, fontweight='bold')
            ax.text(1.5, 1.5, f"FN\n{fn_pct}%", ha='center', va='center', color='red', fontsize=13, fontweight='bold')
            ax.text(0.5, 0.5, f"FP\n{fp_pct}%", ha='center', va='center', color='red', fontsize=13, fontweight='bold')
            ax.text(1.5, 0.5, f"TN\n{tn_pct}%", ha='center', va='center', color='green', fontsize=13, fontweight='bold')
            
            ax.set_xticks([0.5, 1.5])
            ax.set_xticklabels(['Responde', 'No Responde'], fontsize=10)
            ax.xaxis.tick_top()
            
            ax.set_yticks([1.5, 0.5])
            ax.set_yticklabels(['Con Respuesta', 'Sin Respuesta'], fontsize=11)
            ax.set_ylabel('', fontsize=12, labelpad=20)
            
            ax.tick_params(axis='both', which='both', length=0)
            
            ax.text(1.0, -0.35, f"{letter} {model}\n(Exactitud: {accuracy}%)", fontsize=12, fontweight='bold', ha='center', va='top')
            
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_color('gray')
                spine.set_linewidth(1.5)
                
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.6)
    
    filename = "ablation_matrices_confusion.png"
    plt.savefig(os.path.join(cm_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_cumulative_time_plot(df, output_dir):
    print("Generating Cumulative Time Plot...")
    time_dir = os.path.join(output_dir, 'cumulative_times')
    os.makedirs(time_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    for model in MODEL_ORDER:
        model_times = df[df['Model'] == model]['ExecTime'].dropna()
        cumulative_times = model_times.cumsum().values
        x_axis = np.arange(1, len(cumulative_times) + 1)
        me = max(1, len(x_axis) // 15) if len(x_axis) > 0 else 1
        
        plt.plot(x_axis, cumulative_times, label=model, linewidth=2, 
                 color=COLORS.get(model, '#000000'),
                 linestyle=LINE_STYLES.get(model, '-'),
                 marker=MARKERS.get(model, None),
                 markevery=me)
        
    plt.title('Tiempo de Ejecución Acumulado (Estudio Ablación)', fontsize=16, fontweight='bold')
    plt.xlabel('Número de Preguntas Procesadas', fontsize=14)
    plt.ylabel('Tiempo Acumulado (s)', fontsize=14)
    
    plt.legend(title='Modelo', fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    filename = "ablation_cumulative_time_global.png"
    plt.savefig(os.path.join(time_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

def create_distribution_plots(df, output_dir, models_to_plot):
    print("Generating Distribution (Optimized) Plots...")
    dist_dir = os.path.join(output_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)
    
    plot_df = df[df['Model'].isin(models_to_plot)].copy()
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=models_to_plot, ordered=True)
    
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], categories=dataset_order, ordered=True)
    
    metrics = ['ExecTime', 'ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore']
    
    for metric in metrics:
        display_metric = 'Tiempo de Ejecución' if metric == 'ExecTime' else metric
        
        if metric in ['ExactMatch', 'InclusionMatch']:
            plt.figure(figsize=(12, 7))
            ax = sns.barplot(data=plot_df, x='Dataset', y=metric, hue='Model', palette=COLORS, errorbar=None)
            
            for container, model in zip(ax.containers, models_to_plot):
                hatch = HATCHES.get(model, '')
                if hatch:
                    for bar in container:
                        bar.set_hatch(hatch)
            
            plt.title(f'Proporción Media de {display_metric} (Estudio Ablación)', fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Base de Datos', fontsize=14)
            plt.ylabel(f'Media de {display_metric}', fontsize=14)
            
            legend_elements = [mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES[m], label=m) for m in models_to_plot]
            plt.legend(handles=legend_elements, title='Modelo', fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
            plt.tight_layout()
            
            filename = f"ablation_distribution_{metric}.png"
            plt.savefig(os.path.join(dist_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            families = {
                'Distil': ['BiLBERT-Distil (Congelado)', 'BiLBERT-Distil (Completo)'],
                'Large': ['BiLBERT-Large (Congelado)', 'BiLBERT-Large (Completo)']
            }
            
            for family_name, family_models in families.items():
                family_df = plot_df[plot_df['Model'].isin(family_models)].copy()
                family_df['Model'] = pd.Categorical(family_df['Model'], categories=family_models, ordered=True)
                
                if metric in ['ROUGE_L', 'BERTScore']:
                    data_to_plot = family_df[family_df['Status'] == 'TP']
                    title_suffix = " (Solo TP)"
                else:
                    data_to_plot = family_df
                    title_suffix = ""
                    
                g = sns.displot(
                    data=data_to_plot, x=metric, hue='Model', col='Dataset', col_wrap=2,
                    kind='kde', fill=True, alpha=0.3, palette=COLORS, common_norm=False,
                    height=4, aspect=1.5, facet_kws={'sharey': False, 'sharex': False} 
                )
                
                for ax_grid in g.axes.flat:
                    poly_cols = [c for c in ax_grid.collections if isinstance(c, mcollections.PolyCollection)]
                    if poly_cols:
                        poly_cols.sort(key=lambda c: c.get_paths()[0].get_extents().get_points()[:,0].mean())
                        for col, model in zip(poly_cols, family_models):
                            hatch = HATCHES.get(model, '')
                            if hatch:
                                col.set_hatch(hatch)
                
                g.set_titles("{col_name}", size=14, weight='bold') 
                g.set_axis_labels(display_metric, "Densidad")
                g.fig.suptitle(f'Distribución de {display_metric} - Familia {family_name}{title_suffix}', fontsize=16, fontweight='bold', y=1.05)
                
                sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, title="Modelo", frameon=False)
                if g.legend is not None:
                    for patch, text in zip(g.legend.get_patches(), g.legend.get_texts()):
                        model_name = text.get_text()
                        if model_name in HATCHES and HATCHES[model_name]:
                            patch.set_hatch(HATCHES[model_name])
                
                filename = f"ablation_distribution_{metric}_{family_name.lower()}.png"
                plt.savefig(os.path.join(dist_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()

def main():
    sns.set_theme(style="whitegrid")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    completo_file = os.path.join(script_dir, 'tfg_results_mymodels.xlsx') 
    congelado_file = os.path.join(script_dir, 'tfg_results_mymodels_congelado.xlsx') 
    output_directory = os.path.join(script_dir, 'exported_BiL_comparison')
    
    if not os.path.exists(completo_file) or not os.path.exists(congelado_file):
        print("Error: Files not found")
        return

    df = load_ablation_data(completo_file, congelado_file)
    
    create_boxplots(df, output_directory)
    create_summary_tables(df, output_directory)
    create_confusion_matrixes(df, output_directory)
    create_cumulative_time_plot(df, output_directory)
    create_time_summary_table(df, output_directory)
    create_distribution_plots(df, output_directory, models_to_plot=MODEL_ORDER)
    
    print(f"\nGraphs and tables saved in: '{output_directory}'.")

if __name__ == "__main__":
    main()