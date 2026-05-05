import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# 1. NUEVO ORDEN: Definido exactamente como pediste
MODEL_ORDER = [
    'DistilBERT', 
    'SparseDistilBERT', 
    'BiLBERT-Distil', 
    'BERTLarge', 
    'SparseBERTLarge', 
    'BiLBERT-Large'
]

def load_and_combine_data(hist_filepath, new_filepath):
    """
    Carga los datos históricos y los nuevos, filtra los modelos necesarios
    y los combina en un solo DataFrame.
    """
    print(f"Cargando datos históricos desde {hist_filepath}...")
    df_hist = pd.read_excel(hist_filepath)
    
    print(f"Cargando datos nuevos desde {new_filepath}...")
    df_new = pd.read_excel(new_filepath)
    
    # 2. RENOMBRAR: Aseguramos que los nombres coincidan exactamente con MODEL_ORDER
    # Ajusta las claves de estos diccionarios si en tu Excel se llaman diferente
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
    
    # 3. FILTRAR: Sacamos solo los modelos que nos interesan de cada Excel
    df_hist_filtered = df_hist[df_hist['Model'].isin(['DistilBERT', 'BERTLarge'])]
    df_new_filtered = df_new[df_new['Model'].isin(['SparseDistilBERT', 'BiLBERT-Distil', 'SparseBERTLarge', 'BiLBERT-Large'])]
    
    # 4. COMBINAR: Juntamos ambos dataframes
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
            sns.violinplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette='Set2', inner='quartile', density_norm='width', cut=0)
        else:
            plot_data = df
            title_suffix = ""
            sns.boxplot(data=plot_data, x='Dataset', y=metric, hue='Model', palette='Set2', showfliers=True)
        
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
        
        png_filename = f"comparative_boxplot_{metric}.png"
        plt.savefig(os.path.join(box_dir, png_filename), dpi=300)
        plt.close()

    print(f"-> 3 condensed comparative box plots saved in '{box_dir}'.")

def create_summary_tables(df, output_dir):
    print("Generating Summary Tables...")
    metrics_to_mean = ['ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore']    
    datasets = df['Dataset'].unique()
    
    table_dir = os.path.join(output_dir, 'tables')
    os.makedirs(table_dir, exist_ok=True)

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
        
        fig, ax = plt.subplots(figsize=(16, 3))
        ax.axis('off')
        table = ax.table(cellText=summary_df_es.values, 
                         colLabels=summary_df_es.columns, 
                         loc='center', 
                         cellLoc='center')
        
        for (row, col), cell in table.get_celld().items():
            if row == 0: 
                cell.set_text_props(fontweight='bold', fontsize=12)
            else: 
                cell.set_text_props(fontsize=10)
        
        table.auto_set_column_width(col=list(range(len(summary_df_es.columns))))
        
        plt.title(f'Media de Resultados por Modelo en {dataset}', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        png_filename = f"summary_table_{dataset}.png".replace(" ", "_")
        plt.savefig(os.path.join(table_dir, png_filename), dpi=300)
        plt.close()

    print(f"-> Summary tables saved in '{table_dir}'.")

def create_time_summary_table(df, output_dir):
    print("Generating Cross-Dataset Time Summary Table...")
    table_dir = os.path.join(output_dir, 'tables')
    
    time_table = df.pivot_table(index='Model', columns='Dataset', values='ExecTime', aggfunc='mean', observed=False)
    
    dataset_order = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    time_table = time_table.reindex(MODEL_ORDER).reindex(columns=dataset_order).reset_index()
    time_table = time_table.round(5) 

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis('off')
    ax.table(cellText=time_table.values, colLabels=time_table.columns, loc='center', cellLoc='center')
    plt.title('Tiempo Medio de Ejecución (segundos) por Modelo y BBDD', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(table_dir, "time_summary_cross_table.png"), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrixes(df, output_dir):
    print("Generating Confusion Matrixes...")
    models = df['Model'].unique()
    
    cm_dir = os.path.join(output_dir, 'confusion_matrices')
    os.makedirs(cm_dir, exist_ok=True)

    for model in models:
        df_subset = df[df['Model'] == model]
        status_counts = df_subset['Status'].value_counts().to_dict()
        
        tp = int(status_counts.get('TP',0))
        tn = int(status_counts.get('TN',0))
        fp = int(status_counts.get('FP',0))
        fn = int(status_counts.get('FN',0))
        
        cm = np.array([[tp, fn],
                       [fp, tn]], dtype=int)
        
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
    print("Generating Cumulative Time Plot...")
    time_dir = os.path.join(output_dir, 'cumulative_times')
    os.makedirs(time_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 7))
    
    for model in MODEL_ORDER:
        model_times = df[df['Model'] == model]['ExecTime'].dropna()
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

    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        df_subset = df[df['Dataset'] == dataset]
        plt.figure(figsize=(12, 7))
        
        for model in MODEL_ORDER:
            model_times = df_subset[df_subset['Model'] == model]['ExecTime'].dropna()
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
    sns.set_theme(style="whitegrid")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # --- IMPORTANTE: CONFIGURA TUS RUTAS AQUÍ ---
    # Archivo original con la comparativa histórica (donde están DistilBERT y BERTLarge originales)
    hist_file = os.path.join(script_dir, 'tfg_results.xlsx') 
    
    # Archivo nuevo con tus arquitecturas modificadas
    new_file = os.path.join(script_dir, 'tfg_results_mymodels.xlsx') 
    
    # Carpeta exclusiva para que no se mezcle con las gráficas antiguas
    output_directory = os.path.join(script_dir, 'exported_results_mymodels')
    # --------------------------------------------
    
    if not os.path.exists(hist_file):
        print(f"Error: No se encontró el archivo histórico '{hist_file}'.")
        return
    if not os.path.exists(new_file):
        print(f"Error: No se encontró el archivo nuevo '{new_file}'.")
        return

    # Cargamos y combinamos
    df = load_and_combine_data(hist_file, new_file)
    
    # Generamos gráficas
    create_boxplots(df, output_directory)
    create_summary_tables(df, output_directory)
    create_confusion_matrixes(df, output_directory)
    create_cumulative_time_plot(df, output_directory)
    create_time_summary_table(df, output_directory)
    
    print(f"\n¡Proceso completado! Revisa la carpeta '{output_directory}'.")

if __name__ == "__main__":
    main()