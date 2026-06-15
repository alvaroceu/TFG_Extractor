import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.collections as mcollections

MODEL_ORDER = [
    'DistilBERT', 
    'SparseDistilBERT', 
    'BiLBERT-Distil', 
    'BERTLarge', 
    'SparseBERTLarge', 
    'BiLBERT-Large'
]

# Colors
COLORS = {
    'DistilBERT': '#1f77b4',
    'SparseDistilBERT': '#1f77b4',
    'BiLBERT-Distil': '#1f77b4',
    'BERTLarge': '#d62728',
    'SparseBERTLarge': '#d62728',
    'BiLBERT-Large': '#d62728'
}

HATCHES = {
    'DistilBERT': '',
    'SparseDistilBERT': '//',    
    'BiLBERT-Distil': 'xx',      
    'BERTLarge': '',
    'SparseBERTLarge': '//',     
    'BiLBERT-Large': 'xx'        
}

LINE_STYLES = {
    'DistilBERT': '-',
    'SparseDistilBERT': '--',
    'BiLBERT-Distil': ':',
    'BERTLarge': '-',
    'SparseBERTLarge': '--',
    'BiLBERT-Large': ':'
}

MARKERS = {
    'DistilBERT': None,
    'SparseDistilBERT': None,
    'BiLBERT-Distil': 'X',       
    'BERTLarge': None,
    'SparseBERTLarge': None,
    'BiLBERT-Large': 'X'         
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

def load_and_combine_data(hist_filepath, new_filepath):
    print(f"Loading og models data from {hist_filepath}...")
    df_hist = pd.read_excel(hist_filepath)
    
    print(f"Loading my models data from {new_filepath}...")
    df_new = pd.read_excel(new_filepath)
    
    rename_hist = {
        'Transformer DistilBERT': 'DistilBERT',
        'Transformer BERT': 'BERTLarge'
    }
    
    rename_new = {
        'Transformer SparseDistilBERT': 'SparseDistilBERT',
        'Transformer BiLBERTDistil': 'BiLBERT-Distil',
        'Transformer SparseBERTLarge': 'SparseBERTLarge',
        'Transformer BiLBERTLarge': 'BiLBERT-Large',
    }
    
    df_hist['Model'] = df_hist['Model'].replace(rename_hist)
    df_new['Model'] = df_new['Model'].replace(rename_new)
    
    df_hist_filtered = df_hist[df_hist['Model'].isin(['DistilBERT', 'BERTLarge'])]
    df_new_filtered = df_new[df_new['Model'].isin(['SparseDistilBERT', 'BiLBERT-Distil', 'SparseBERTLarge', 'BiLBERT-Large'])]
    
    df_combined = pd.concat([df_hist_filtered, df_new_filtered], ignore_index=True)
    return df_combined

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

        plt.title(f'Distribución Comparativa de {display_metric} por Base de Datos y Modelo{title_suffix}', fontsize=16, fontweight='bold')
        plt.xlabel('Base de Datos', fontsize=14)
        plt.xticks(fontsize=12, rotation=0) 
        plt.yticks(fontsize=12)
        
        legend_elements = [mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES[m], label=m) for m in MODEL_ORDER]
        ax.legend(handles=legend_elements, title='Modelo', fontsize=11, title_fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout()
        
        png_filename = f"comparative_boxplot_{metric}.png"
        plt.savefig(os.path.join(box_dir, png_filename), dpi=300, bbox_inches='tight')
        plt.close()

    print(f"-> 3 condensed comparative box plots saved in '{box_dir}'.")

def create_summary_tables(df, output_dir):
    print("Generating Summary Tables (Excel)...")
    metrics_to_mean = ['ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore']    
    datasets = df['Dataset'].unique()
    
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)
    excel_path = os.path.join(table_dir, 'summary_tables.xlsx')

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

    print(f"-> Summary tables saved in '{excel_path}'.")

def create_time_summary_table(df, output_dir):
    print("Generating Time Summary Table with Bootstrap 95% CI (Modified)...")
    from scipy import stats
    
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)

    excel_path = os.path.join(table_dir, 'time_summary_cross_table.xlsx')
    
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    model_order = [
        'DistilBERT', 'SparseDistilBERT', 'BiLBERT-Distil', 
        'BERTLarge', 'SparseBERTLarge', 'BiLBERT-Large'
    ]
    
    results = []
    
    for model in model_order:
        row_data = {'Modelo': model}
        for dataset in dataset_order:
            data = df[(df['Model'] == model) & (df['Dataset'] == dataset)]['ExecTime'].dropna().values
            
            if len(data) > 1:
                mean_val = np.mean(data)
                # 95% Cobnfidence interval
                res = stats.bootstrap((data,), np.mean, confidence_level=0.95, random_state=42)
                ci_l, ci_u = res.confidence_interval
                
                row_data[dataset] = f"{mean_val:.5f} [{ci_l:.5f}, {ci_u:.5f}]"
            else:
                row_data[dataset] = "-"
                
        results.append(row_data)
        
    summary_df = pd.DataFrame(results)
    summary_df = summary_df[['Modelo'] + dataset_order]
    
    summary_df.to_excel(excel_path, index=False, sheet_name='Tiempos_Bootstrap')
    print(f"-> Time summary table saved in '{excel_path}'.")

def create_confusion_matrixes(df, output_dir):
    print("Generating Composite Confusion Matrix (Modified Models)...")
    
    matrix_order = [
        ['DistilBERT', 'SparseDistilBERT', 'BiLBERT-Distil'],
        ['BERTLarge', 'SparseBERTLarge', 'BiLBERT-Large']
    ]
    letters = [['a)', 'b)', 'c)'], ['d)', 'e)', 'f)']]
    
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle('Matrices de Confusión de Modelos Modificados', fontsize=18, fontweight='bold', y=1.05)
    
    for row in range(2):
        for col in range(3):
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
            ax.set_xticklabels(['Responde', 'No Responde'], fontsize=11)
            ax.xaxis.tick_top()
            
            ax.set_yticks([1.5, 0.5])
            ax.set_yticklabels(['Con Respuesta', 'Sin Respuesta'], fontsize=11)
            ax.set_ylabel('', fontsize=12, labelpad=20)
            
            ax.tick_params(axis='both', which='both', length=0)
            
            ax.text(1.0, -0.35, f"{letter} {model} (Exactitud: {accuracy}%)", fontsize=14, fontweight='bold', ha='center', va='top')
            
            ax.set_facecolor('white')
            for spine in ax.spines.values():
                spine.set_color('gray')
                spine.set_linewidth(1.5)
                
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.6, wspace=0.8)
    
    filename = "matrices_confusion_modificados.png"
    plt.savefig(os.path.join(cm_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Composite confusion matrix saved as '{filename}' in '{cm_dir}'.")

def create_cumulative_time_plot(df, output_dir):
    print("Generating Cumulative Time Plot...")
    time_dir = os.path.join(output_dir, 'cumulative_times')
    os.makedirs(time_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
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
        
    plt.title('Tiempo de Ejecución Acumulado por Modelo (Global)', fontsize=16, fontweight='bold')
    plt.xlabel('Número de Preguntas Procesadas', fontsize=14)
    plt.ylabel('Tiempo Acumulado (s)', fontsize=14)
    
    plt.legend(title='Modelo', fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    filename = "cumulative_time_global.png"
    plt.savefig(os.path.join(time_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        df_subset = df[df['Dataset'] == dataset]
        plt.figure(figsize=(12, 7))
        
        for model in MODEL_ORDER:
            model_times = df_subset[df_subset['Model'] == model]['ExecTime'].dropna()
            cumulative_times = model_times.cumsum().values

            x_axis = np.arange(1, len(cumulative_times) + 1)
            me = max(1, len(x_axis) // 15) if len(x_axis) > 0 else 1
            
            plt.plot(x_axis, cumulative_times, label=model, linewidth=2, 
                     color=COLORS.get(model, '#000000'),
                     linestyle=LINE_STYLES.get(model, '-'),
                     marker=MARKERS.get(model, None),
                     markevery=me)
                
        plt.title(f'Tiempo de Ejecución Acumulado por Modelo en {dataset}', fontsize=16, fontweight='bold')
        plt.xlabel('Número de Preguntas Procesadas', fontsize=14)
        plt.ylabel('Tiempo Acumulado (s)', fontsize=14)
        
        plt.legend(title='Modelo', fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        filename = f"cumulative_time_{dataset}.png".replace(" ", "_")
        plt.savefig(os.path.join(time_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def create_distribution_plots(df, output_dir, models_to_plot):
    """
    Distribution plots (grouped for discrete metrics and KDE for continuous) for each BBDD
    """
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
        
        # Discrete metrics
        if metric in ['ExactMatch', 'InclusionMatch']:
            plt.figure(figsize=(12, 7))
            ax = sns.barplot(
                data=plot_df, 
                x='Dataset', 
                y=metric, 
                hue='Model', 
                palette=COLORS, 
                errorbar=None
            )
            
            for container, model in zip(ax.containers, models_to_plot):
                hatch = HATCHES.get(model, '')
                if hatch:
                    for bar in container:
                        bar.set_hatch(hatch)
            
            plt.title(f'Proporción Media de {display_metric} por BBDD y Modelo', fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Base de Datos', fontsize=14)
            plt.ylabel(f'Media de {display_metric}', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            legend_elements = [mpatches.Patch(facecolor=COLORS[m], hatch=HATCHES[m], label=m) for m in models_to_plot]
            plt.legend(handles=legend_elements, title='Modelo', fontsize=11, title_fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
            plt.tight_layout()
            
            filename = f"distribution_{metric}.png"
            plt.savefig(os.path.join(dist_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        
        #Continuous metrics
        else:
            families = {
                'Distil': ['DistilBERT', 'SparseDistilBERT', 'BiLBERT-Distil'],
                'Large': ['BERTLarge', 'SparseBERTLarge', 'BiLBERT-Large']
            }
            
            for family_name, family_models in families.items():
                family_df = plot_df[plot_df['Model'].isin(family_models)].copy()
                family_df['Model'] = pd.Categorical(family_df['Model'], categories=family_models, ordered=True)
                
                if metric in ['ROUGE_L', 'BERTScore']:
                    data_to_plot = family_df[family_df['Status'] == 'TP']
                    title_suffix = " (Solo Verdaderos Positivos)"
                else:
                    data_to_plot = family_df
                    title_suffix = ""
                    
                g = sns.displot(
                    data=data_to_plot, 
                    x=metric, 
                    hue='Model', 
                    col='Dataset',   
                    col_wrap=2,      
                    kind='kde',      
                    fill=True, 
                    alpha=0.3,       
                    palette=COLORS, 
                    common_norm=False,
                    height=4,        
                    aspect=1.5,      
                    facet_kws={'sharey': False, 'sharex': False} 
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
                
                g.fig.suptitle(f'Distribución de {display_metric} por BBDD - Familia {family_name}{title_suffix}', fontsize=16, fontweight='bold', y=1.05)
                
                sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.05), ncol=3, title="Modelo", frameon=False)
                plt.setp(g.legend.get_texts(), fontsize='11')
                
                if g.legend is not None:
                    for patch, text in zip(g.legend.get_patches(), g.legend.get_texts()):
                        model_name = text.get_text()
                        if model_name in HATCHES and HATCHES[model_name]:
                            patch.set_hatch(HATCHES[model_name])
                
                filename = f"distribution_{metric}_{family_name.lower()}.png"
                plt.savefig(os.path.join(dist_dir, filename), dpi=300, bbox_inches='tight')
                plt.close()
        
    print(f"-> Distribution plots saved in '{dist_dir}'.")

def main():
    sns.set_theme(style="whitegrid")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    hist_file = os.path.join(script_dir, 'tfg_results.xlsx') 
    new_file = os.path.join(script_dir, 'tfg_results_mymodels.xlsx') 
    output_directory = os.path.join(script_dir, 'exported_results_mymodels')
    
    if not os.path.exists(hist_file):
        print(f"Error: The file '{hist_file}' does not exist.")
        return
    if not os.path.exists(new_file):
        print(f"Error: The file '{new_file}' does not exist.")
        return

    df = load_and_combine_data(hist_file, new_file)
    
    create_boxplots(df, output_directory)
    create_summary_tables(df, output_directory)
    create_confusion_matrixes(df, output_directory)
    create_cumulative_time_plot(df, output_directory)
    create_time_summary_table(df, output_directory)
    create_distribution_plots(df, output_directory, models_to_plot=MODEL_ORDER)
    
    print(f"\n¡Process completed! Check the '{output_directory}' folder.")

if __name__ == "__main__":
    main()