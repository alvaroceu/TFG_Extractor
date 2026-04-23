import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

MODEL_ORDER = ['BoW', 'tf-idf', 'gloVe', 'UseDan', 'DistilBERT', 'BERTLarge']

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
            sns.violinplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette='Set2', inner='quartile', density_norm='width', cut=0)
        else:
            plot_data = df
            title_suffix = ""
            sns.boxplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette='Set2', showfliers=True)
        
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
        plt.legend(title='Modelo', fontsize=10, title_fontsize=12, loc='upper left')
        plt.tight_layout()
        
        # Save image
        png_filename = f"comparative_boxplot_{metric}.png"
        plt.savefig(os.path.join(box_dir, png_filename), dpi=300)
        
        plt.close()

    print(f"-> 3 condensed comparative box plots saved in '{box_dir}'.")

def create_summary_tables(df, output_dir):
    """
    Generates summary tables for each dataset, calculating metric means per model.
    """

    print("Generating Summary Tables...")
    metrics_to_mean = ['ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore']    
    datasets = df['Dataset'].unique()
    
    # Create subfolder for summary tables
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)

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
        
        # Create table
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.axis('off')
        table = ax.table(cellText=summary_df_es.values, 
                                colLabels=summary_df_es.columns, 
                                loc='center', 
                                cellLoc='center')
        
        for (row, col), cell in table.get_celld().items():
            if row == 0: # Header row
                cell.set_text_props(fontweight='bold', fontsize=12)
            else: # Data rows
                cell.set_text_props(fontsize=10)
        
        table.auto_set_column_width(col=list(range(len(summary_df_es.columns))))
        
        plt.title(f'Media de Resultados por Modelo en {dataset}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        # Save image
        png_filename = f"summary_table_{dataset}.png".replace(" ", "_")
        plt.savefig(os.path.join(table_dir, png_filename), dpi=300)
        plt.close()

    print(f"-> Summary tables saved in '{table_dir}'.")

def create_time_summary_table(df, output_dir):
    """
    Creates a table summarizing the avg times foir each model across all datasets
    """
    print("Generating Cross-Dataset Time Summary Table...")

    # Create subfolder for summary tables
    table_dir = os.path.join(output_dir, 'tables')
    
    # Pivot table for mean times
    time_table = df.pivot_table(index='Model', columns='Dataset', values='ExecTime', aggfunc='mean', observed=False)
    
    # Reorder according to your specific needs
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    time_table = time_table.reindex(MODEL_ORDER).reindex(columns=dataset_order).reset_index()
    time_table = time_table.round(5) 

    # Save as PNG
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    ax.table(cellText=time_table.values, colLabels=time_table.columns, loc='center', cellLoc='center')
    plt.title('Tiempo Medio de Ejecución (segundos) por Modelo y BBDD', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(table_dir, "time_summary_cross_table.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrixes(df, output_dir):
    """
    Generates Confusion Matrixes (TP, TN, FP, FN).
    """

    print("Generating Confusion Matrixes...")
    models = df['Model'].unique()
    
    # Create subfolder for confusion matrixes
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    # Generate a confusion matrix per model combining all datasets
    for model in models:
        df_subset = df[df['Model'] == model]
        
        # Count the occurrences of each status
        status_counts = df_subset['Status'].value_counts().to_dict()
        
        # Extract counts safely
        tp = int(status_counts.get('TP',0))
        tn = int(status_counts.get('TN',0))
        fp = int(status_counts.get('FP',0))
        fn = int(status_counts.get('FN',0))
        
        # Construct the 2x2 confusion matrix array
        # Row 1: Actual Positives (TP, FN)
        # Row 2: Actual Negatives (FP, TN)
        cm = np.array([[tp, fn],
                       [fp, tn]], dtype=int)
        
        # Normalize the matrix
        total = cm.sum()
        if total > 0:
            cm_normalized = cm / total
        else:
            cm_normalized = cm

        plt.figure(figsize=(6, 5))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=['Responde', 'No Responde'],
                    yticklabels=['Con Respuesta', 'Sin Respuesta'])
        
        plt.title(f'Matriz de Confusión de {model}')
        plt.xlabel('Predicción del Modelo')
        plt.ylabel('Verdad Base (Ground Truth)')
        plt.tight_layout()
        
        filename = f"confusion_matrix_{model}.png".replace(" ", "_")
        plt.savefig(os.path.join(cm_dir, filename), dpi=300)
        plt.close()

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
        plt.plot(x_axis, cumulative_times, label=model, linewidth=2)
        
    plt.title('Tiempo de Ejecución Acumulado por Modelo (Global)', fontsize=16, fontweight='bold')
    plt.xlabel('Número de Preguntas Procesadas', fontsize=14)
    plt.ylabel('Tiempo Acumulado (s)', fontsize=14)
    plt.legend(title='Modelo', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    
    filename = "cumulative_time_global.png"
    plt.savefig(os.path.join(time_dir, filename), dpi=300)
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
            plt.plot(x_axis, cumulative_times, label=model, linewidth=2)
                
        plt.title(f'Tiempo de Ejecución Acumulado por Modelo en {dataset}', fontsize=16, fontweight='bold')
        plt.xlabel('Número de Preguntas Procesadas', fontsize=14)
        plt.ylabel('Tiempo Acumulado (s)', fontsize=14)
        plt.legend(title='Modelo', fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.tight_layout()
        
        filename = f"cumulative_time_{dataset}.png".replace(" ", "_")
        plt.savefig(os.path.join(time_dir, filename), dpi=300)
        plt.close()

def main():
    # 1. Set global visual theme for seaborn
    sns.set_theme(style="whitegrid")
    
    # 2. Define input file and output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(script_dir, 'tfg_results.xlsx')
    output_directory = os.path.join(script_dir, 'exported_results')
    
    # Check if the file exists before proceeding
    if not os.path.exists(input_file):
        print(f"Error: The file '{input_file}' was not found in the current directory.")
        return

    # 3. Execute the workflow
    df = load_data(input_file)
    
    create_boxplots(df, output_directory)
    create_summary_tables(df, output_directory)
    create_confusion_matrixes(df, output_directory)
    create_cumulative_time_plot(df, output_directory)
    create_time_summary_table(df, output_directory)
    
    print(f"\nProcess completed! Check the '{output_directory}' folder.")

if __name__ == "__main__":
    main()