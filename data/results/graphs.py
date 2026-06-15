import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

MODEL_ORDER = ['BoW', 'tf-idf', 'gloVe', 'UseDan', 'DistilBERT', 'BERTLarge']

# Colors
COLORS = {
    'BoW': '#2ca02c',        
    'tf-idf': '#ff7f0e',     
    'gloVe': '#9467bd',      
    'UseDan': '#8c564b',     
    'DistilBERT': '#1f77b4', 
    'BERTLarge': '#d62728'   
}

def load_data(filepath):
    """
    Loads the dataset from an Excel file.
    """

    print(f"Loading data from {filepath}...")
    df = pd.read_excel(filepath)
    
    rename = {
        'Embeddings gloVe': 'gloVe',
        'UseDanLSTM': 'UseDan',
        'Transformer DistilBERT': 'DistilBERT',
        'Transformer BERT': 'BERTLarge'
    }
    df['Model'] = df['Model'].replace(rename)
    return df

def create_boxplots(df, output_dir):
    """
    Generates condensed box plots for key numerical metrics.
    """

    print("Generating Box Plots...")
    metrics = ['ExecTime', 'ROUGE_L', 'BERTScore']
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    
    # Create subfolder for box plots
    box_dir = os.path.join(output_dir, 'box_plots')
    os.makedirs(box_dir, exist_ok=True)

    # Convert columns to ordered Categorical to enforce visual order in plots
    df['Dataset'] = pd.Categorical(df['Dataset'], categories=dataset_order, ordered=True)
    df['Model'] = pd.Categorical(df['Model'], categories=MODEL_ORDER, ordered=True)

    # Iterate through metrics
    for metric in metrics:

        plt.figure(figsize=(14, 8))
        display_metric = 'Tiempo de Ejecución' if metric == 'ExecTime' else metric
        
        # Filter by TP for ROUGE_L and BERTScore
        if metric in ['ROUGE_L', 'BERTScore']:
            plot_data = df[df['Status'] == 'TP']
            title_suffix = " (Solo Verdaderos Positivos)"
            sns.violinplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette=COLORS, inner='quartile', density_norm='width', cut=0)
        else:
            plot_data = df
            title_suffix = ""
            sns.boxplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette=COLORS, showfliers=True)
        
        # Apply Log-Scale for ExecTime to improve visibility
        if metric == 'ExecTime':
            plt.yscale('log')
            plt.ylabel(f'{display_metric} (Escala Logarítmica)', fontsize=14)
        else:
            plt.ylabel(f'{display_metric}', fontsize=14)

        plt.title(f'Distribución Comparativa de {display_metric} por Base de Datos y Modelo{title_suffix}', fontsize=16, fontweight='bold')
        plt.xlabel('Base de Datos', fontsize=14)
        plt.xticks(fontsize=12, rotation=0) 
        plt.yticks(fontsize=12)
        
        plt.legend(title='Modelo', fontsize=11, title_fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
        plt.tight_layout()
        
        # Save image 
        png_filename = f"comparative_boxplot_{metric}.png"
        plt.savefig(os.path.join(box_dir, png_filename), dpi=300, bbox_inches='tight')
        
        plt.close()

    print(f"-> 3 condensed comparative box plots saved in '{box_dir}'.")

def create_summary_tables(df, output_dir):
    """
    Generates summary tables for each dataset, calculating metric means per model and saving to Excel.
    """

    print("Generating Summary Tables (Excel)...")
    metrics_to_mean = ['ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore']    
    datasets = df['Dataset'].unique()
    
    # Create subfolder for summary tables
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)
    excel_path = os.path.join(table_dir, 'summary_tables.xlsx')

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for dataset in datasets:
            df_subset = df[df['Dataset'] == dataset]
            
            # Global metrics
            all_means = df_subset.groupby('Model', observed=False)[metrics_to_mean].mean().reset_index()
            all_means.columns = ['Model'] + [f"{c} (Global)" for c in metrics_to_mean]
            
            # TP metrics
            df_tp = df_subset[df_subset['Status'] == 'TP']
            tp_means = df_tp.groupby('Model', observed=False)[metrics_to_mean].mean().reset_index()
            tp_means.columns = ['Model'] + [f"{c} (TP)" for c in metrics_to_mean]
            
            # Merge my model
            summary_df = pd.merge(all_means, tp_means, on='Model')

            summary_df = summary_df.round(5)
            summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=MODEL_ORDER, ordered=True)
            summary_df = summary_df.sort_values('Model').reset_index(drop=True)
            
            summary_df_es = summary_df.rename(columns={
                'Model': 'Modelo',
            })
            
            # Save to an Excel tab
            sheet_name = str(dataset)[:31] 
            summary_df_es.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"-> Summary tables saved in '{excel_path}'.")

def create_time_summary_table(df, output_dir):
    print("Generating Time Summary Table with Bootstrap 95% CI (Originals)...")
    from scipy import stats
    
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)
    excel_path = os.path.join(table_dir, 'time_summary_cross_table.xlsx')
    
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    model_order = ['BoW', 'tf-idf', 'gloVe', 'UseDan', 'DistilBERT', 'BERTLarge']
    
    results = []
    
    for model in model_order:
        row_data = {'Modelo': model}
        for dataset in dataset_order:
            data = df[(df['Model'] == model) & (df['Dataset'] == dataset)]['ExecTime'].dropna().values
            
            if len(data) > 1:
                mean_val = np.mean(data)
                # 95% Confidence interval
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
    print("Generating Composite Confusion Matrix (Original Models)...")
    
    matrix_order = [
        ['BoW', 'tf-idf', 'gloVe'],
        ['UseDan', 'DistilBERT', 'BERTLarge']
    ]
    letters = [['a)', 'b)', 'c)'], ['d)', 'e)', 'f)']]
    
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    fig.suptitle('Matrices de Confusión de Modelos Originales', fontsize=18, fontweight='bold', y=1.05)
    
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
    
    filename = "matrices_confusion_originales.png"
    plt.savefig(os.path.join(cm_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"-> Composite confusion matrix saved as '{filename}' in '{cm_dir}'.")

def create_cumulative_time_plot(df, output_dir):
    """
    Generates a line plot showing the cumulative execution time for each model 
    as the number of processed questions increases.
    """

    print("Generating Cumulative Time Plot...")
    time_dir = os.path.join(output_dir, 'cumulative_times')
    os.makedirs(time_dir, exist_ok=True)
    
    # Global time plot
    plt.figure(figsize=(12, 7))
    
    for model in MODEL_ORDER:
        # Extract execution times for the specific model
        model_times = df[df['Model'] == model]['ExecTime'].dropna()
        # Calculate cumulative sum
        cumulative_times = model_times.cumsum().values

        x_axis = np.arange(1, len(cumulative_times) + 1)
        plt.plot(x_axis, cumulative_times, label=model, linewidth=2, color=COLORS.get(model, '#000000'))
        
    plt.title('Tiempo de Ejecución Acumulado por Modelo (Global)', fontsize=16, fontweight='bold')
    plt.xlabel('Número de Preguntas Procesadas', fontsize=14)
    plt.ylabel('Tiempo Acumulado (s)', fontsize=14)
    
    plt.legend(title='Modelo', fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=6)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    filename = "cumulative_time_global.png"
    plt.savefig(os.path.join(time_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    # Time plots for each dataset
    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        df_subset = df[df['Dataset'] == dataset]
        
        plt.figure(figsize=(12, 7))
        
        for model in MODEL_ORDER:
            # Extract execution times for the specific model and dataset
            model_times = df_subset[df_subset['Model'] == model]['ExecTime'].dropna()
            # Cumulative sum
            cumulative_times = model_times.cumsum().values

            x_axis = np.arange(1, len(cumulative_times) + 1)
            plt.plot(x_axis, cumulative_times, label=model, linewidth=2, color=COLORS.get(model, '#000000'))
                
        plt.title(f'Tiempo de Ejecución Acumulado por Modelo en {dataset}', fontsize=16, fontweight='bold')
        plt.xlabel('Número de Preguntas Procesadas', fontsize=14)
        plt.ylabel('Tiempo Acumulado (s)', fontsize=14)
        
        plt.legend(title='Modelo', fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=6)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        filename = f"cumulative_time_{dataset}.png".replace(" ", "_")
        plt.savefig(os.path.join(time_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()

def create_distribution_plots(df, output_dir, models_to_plot):
    """
    Distribution plots (grouped for discrete metrics and KDE for continuous) 2x2 subplots for each BBDD
    """
    print("Generating Distribution Plots...")
    dist_dir = os.path.join(output_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)
    
    # Filer models
    plot_df = df[df['Model'].isin(models_to_plot)].copy()
    plot_df['Model'] = pd.Categorical(plot_df['Model'], categories=models_to_plot, ordered=True)
    
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    plot_df['Dataset'] = pd.Categorical(plot_df['Dataset'], categories=dataset_order, ordered=True)
    
    metrics = ['ExecTime', 'ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore']
    
    for metric in metrics:
        display_metric = 'Tiempo de Ejecución' if metric == 'ExecTime' else metric
        
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
            
            plt.title(f'Proporción Media de {display_metric} por BBDD y Modelo', fontsize=16, fontweight='bold', pad=15)
            plt.xlabel('Base de Datos', fontsize=14)
            plt.ylabel(f'Media de {display_metric}', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            
            plt.legend(title='Modelo', fontsize=11, title_fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=6)
            plt.tight_layout()
            
            filename = f"distribution_{metric}.png"
            plt.savefig(os.path.join(dist_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            if metric in ['ROUGE_L', 'BERTScore']:
                data_to_plot = plot_df[plot_df['Status'] == 'TP']
                title_suffix = " (Solo Verdaderos Positivos)"
            else:
                data_to_plot = plot_df
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
            
            g.set_titles("{col_name}", size=14, weight='bold') 
            g.set_axis_labels(display_metric, "Densidad")
            
            g.fig.suptitle(f'Distribución de {display_metric} por BBDD{title_suffix}', fontsize=16, fontweight='bold', y=1.05)
            
            sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, -0.05), ncol=6, title="Modelo")
            plt.setp(g.legend.get_texts(), fontsize='11')
            
            filename = f"distribution_{metric}.png"
            plt.savefig(os.path.join(dist_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        
    print(f"-> Distribution plots saved in '{dist_dir}'.")

def main():
    sns.set_theme(style="whitegrid")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'tfg_results.xlsx')
    output_directory = os.path.join(script_dir, 'exported_results')
    
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found in the current directory.")
        return

    df = load_data(input_file)
    
    create_boxplots(df, output_directory)
    create_summary_tables(df, output_directory)
    create_confusion_matrixes(df, output_directory)
    create_cumulative_time_plot(df, output_directory)
    create_time_summary_table(df, output_directory)
    create_distribution_plots(df, output_directory, models_to_plot=['DistilBERT', 'BERTLarge'])
    
    print(f"\nProcess completed! Check the '{output_directory}' folder.")

if __name__ == "__main__":
    main()