import pandas as pd
import os
from scipy import stats
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings

# Suprimimos los avisos de Wilcoxon sobre empates (normales en ExactMatch)
warnings.filterwarnings("ignore")

PAIRS_TO_COMPARE = [
    ('BERTLarge', 'SparseBERTLarge'),
    ('BERTLarge', 'BiLBERT-Large'),
    ('DistilBERT', 'SparseDistilBERT'),
    ('DistilBERT', 'BiLBERT-Distil'),
    ('DistilBERT', 'BERTLarge'),
    ('SparseDistilBERT', 'SparseBERTLarge'),
    ('BiLBERT-Distil', 'BiLBERT-Large')
]

METRICS = ['ExactMatch', 'InclusionMatch', 'ROUGE_L', 'BERTScore', 'ExecTime']
DATASET_ORDER = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
LETTERS = ['a)', 'b)', 'c)', 'd)']

def load_and_combine_data(hist_filepath, new_filepath):
    print(f"Cargando datos desde Excel...")
    df_hist = pd.read_excel(hist_filepath)
    df_new = pd.read_excel(new_filepath)
    
    rename_hist = {'Transformer DistilBERT': 'DistilBERT', 'Transformer BERT': 'BERTLarge'}
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
    
    return pd.concat([df_hist_filtered, df_new_filtered], ignore_index=True)

def safe_to_numeric(series):
    if series.dtype == object:
        series = series.astype(str).str.replace(',', '.')
    return pd.to_numeric(series, errors='coerce')

def run_tests_and_qqplots(df, output_dir):
    print("Iniciando análisis estadístico (Wilcoxon) y recolección de datos para Q-Q Plots...")
    datasets = df['Dataset'].unique()
    scopes = ['Global', 'TP']
    
    all_models_in_pairs = set([m for pair in PAIRS_TO_COMPARE for m in pair])
    
    qq_dir = os.path.join(output_dir, 'qq_plots')
    os.makedirs(qq_dir, exist_ok=True)

    results = []
    
    global_qq_data = {metric: {model: {} for model in all_models_in_pairs} for metric in METRICS}

    for dataset in datasets:
        df_dataset = df[df['Dataset'] == dataset]

        for scope in scopes:
            for metric in METRICS:
                if scope == 'TP' and metric == 'ExecTime':
                    continue

                # =========================================================
                # FASE 1: EVALUACIÓN DE NORMALIDAD (R^2)
                # =========================================================
                normality_cache = {}
                
                for model in all_models_in_pairs:
                    df_model = df_dataset[df_dataset['Model'] == model][['Question ID', 'Status', metric]]
                    df_model[metric] = safe_to_numeric(df_model[metric])
                    df_model = df_model.dropna(subset=[metric])
                    
                    if scope == 'TP':
                        df_model = df_model[df_model['Status'] == 'TP']
                        
                    data = df_model[metric].values
                    
                    if len(data) > 10:
                        (osm, osr), (slope, intercept, r_value) = stats.probplot(data, dist="norm")
                        r_squared = r_value ** 2
                        
                        if scope == 'Global':
                            global_qq_data[metric][model][dataset] = (data, r_squared)
                        
                        normality_cache[model] = r_squared
                    else:
                        normality_cache[model] = np.nan

                # =========================================================
                # FASE 2: PRUEBA DE WILCOXON SIEMPRE
                # =========================================================
                for model_a, model_b in PAIRS_TO_COMPARE:
                    df_a = df_dataset[df_dataset['Model'] == model_a][['Question ID', 'Status', metric]]
                    df_b = df_dataset[df_dataset['Model'] == model_b][['Question ID', 'Status', metric]]
                    
                    merged = pd.merge(df_a, df_b, on='Question ID', suffixes=('_A', '_B'))
                    merged[f'{metric}_A'] = safe_to_numeric(merged[f'{metric}_A'])
                    merged[f'{metric}_B'] = safe_to_numeric(merged[f'{metric}_B'])
                    merged = merged.dropna(subset=[f'{metric}_A', f'{metric}_B'])

                    if scope == 'TP':
                        merged = merged[(merged['Status_A'] == 'TP') & (merged['Status_B'] == 'TP')]

                    data_a = merged[f'{metric}_A'].values
                    data_b = merged[f'{metric}_B'].values

                    if len(data_a) > 10:
                        r2_a = normality_cache.get(model_a, np.nan)
                        r2_b = normality_cache.get(model_b, np.nan)
                        
                        try:
                            res = stats.wilcoxon(data_a, data_b, alternative='two-sided', method='approx')
                            p_val = res.pvalue
                            z_stat = res.zstatistic
                            effect_size_r = abs(z_stat) / np.sqrt(len(data_a))
                        except ValueError:
                            p_val = np.nan
                            effect_size_r = np.nan

                        median_a, median_b = np.median(data_a), np.median(data_b)
                        iqr_a, iqr_b = stats.iqr(data_a), stats.iqr(data_b)
                        mean_a, mean_b = np.mean(data_a), np.mean(data_b)
                        std_a, std_b = np.std(data_a, ddof=1), np.std(data_b, ddof=1)

                        results.append({
                            'Dataset': dataset,
                            'Numero preguntas': len(data_a),
                            'Caso': scope,
                            'Metrica': metric,
                            'Modelo A': model_a,
                            'Modelo B': model_b,
                            'Mediana A': round(median_a, 4),
                            'IQR A': round(iqr_a, 4),
                            'Media A': round(mean_a, 4),
                            'Std A': round(std_a, 4),
                            'Mediana B': round(median_b, 4),
                            'IQR B': round(iqr_b, 4),
                            'Media B': round(mean_b, 4),
                            'Std B': round(std_b, 4),
                            'R^2 Q-Q Plot (A)': round(r2_a, 4) if pd.notna(r2_a) else 'N/A',
                            'R^2 Q-Q Plot (B)': round(r2_b, 4) if pd.notna(r2_b) else 'N/A',
                            'Wilcoxon p-val': p_val,
                            'Tamaño efecto r': round(effect_size_r, 3) if pd.notna(effect_size_r) else 'N/A',
                            'Significativo (p<0.05)': 'Sí' if pd.notna(p_val) and p_val < 0.05 else 'No'
                        })

    # =========================================================
    # FASE 3: GENERACIÓN DE MATRICES 1x4 PARA Q-Q PLOTS
    # =========================================================
    print("Generando matrices 1x4 de Q-Q Plots...")
    for metric in METRICS:
        for model in all_models_in_pairs:
            # Crear figura con 1 fila y 4 columnas (formato horizontal)
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            axes = axes.flatten() 
            
            fig.suptitle(f'Q-Q Plots: {model} - {metric}', fontsize=16, fontweight='bold', y=1.05)
            
            for i, dataset in enumerate(DATASET_ORDER):
                ax = axes[i]
                if dataset in global_qq_data[metric][model]:
                    data, r2 = global_qq_data[metric][model][dataset]
                    sm.qqplot(data, line='45', fit=True, ax=ax)
                    ax.set_title(f"{LETTERS[i]} {dataset} (R²: {r2:.4f})", fontsize=12, fontweight='bold')
                else:
                    ax.axis('off')
                    ax.set_title(f"{LETTERS[i]} {dataset} (Sin datos)", fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            
            filename = f"{model}_{metric}.png".replace(' ', '_')
            plt.savefig(os.path.join(qq_dir, filename), dpi=200, bbox_inches='tight')
            plt.close(fig)

    results_df = pd.DataFrame(results)
    
    results_df['Resumen Wilcoxon p'] = results_df['Wilcoxon p-val'].apply(
        lambda x: '< 0.0001' if pd.notna(x) and (x < 0.0001 or x == 0.0) else (f"{x:.4f}" if pd.notna(x) else "N/A")
    )
    
    cols = [
        'Dataset', 'Numero preguntas', 'Caso', 'Metrica', 
        'Modelo A', 'Modelo B', 
        'Mediana A', 'IQR A', 'Media A', 'Std A',
        'Mediana B', 'IQR B', 'Media B', 'Std B',
        'R^2 Q-Q Plot (A)', 'R^2 Q-Q Plot (B)',
        'Wilcoxon p-val', 'Resumen Wilcoxon p', 'Tamaño efecto r', 'Significativo (p<0.05)'
    ]
    results_df = results_df[cols]

    # =========================================================
    # FASE 4: CREACIÓN DE LA TABLA RESUMEN DE EFECTOS
    # =========================================================
    print("Generando tabla resumen de tamaños de efecto...")
    summary_records = []
    
    metric_mapping = [
        ('ROUGE_L', 'ROUGE-L (r)'),
        ('BERTScore', 'BERTScore (r)'),
        ('ExactMatch', 'ExactMatch (r)'),
        ('InclusionMatch', 'InclusionMatch (r)'),
        ('ExecTime', 'Tiempo de Ejecución (r)')
    ]
    
    for model_a, model_b in PAIRS_TO_COMPARE:
        record = {'Par': f"{model_a} vs {model_b}"}
        
        for metric_internal, metric_display in metric_mapping:
            # Filtrar todas las ocurrencias de esta pareja y métrica
            mask = (results_df['Modelo A'] == model_a) & \
                   (results_df['Modelo B'] == model_b) & \
                   (results_df['Metrica'] == metric_internal)
            
            r_values = results_df.loc[mask, 'Tamaño efecto r']
            r_values = pd.to_numeric(r_values, errors='coerce').dropna()
            
            if r_values.empty:
                record[metric_display] = "-"
            else:
                min_r = r_values.min()
                max_r = r_values.max()
                
                # Formatear el rango con comas para español
                if np.isclose(min_r, max_r):
                    record[metric_display] = f"{min_r:.2f}".replace('.', ',')
                else:
                    record[metric_display] = f"{min_r:.2f} - {max_r:.2f}".replace('.', ',')
                    
        record['Efecto'] = ""  # Columna vacía para rellenar manualmente
        summary_records.append(record)

    summary_df = pd.DataFrame(summary_records)

    # GUARDADO EN EXCEL
    excel_path = os.path.join(output_dir, 'estadisticas_wilcoxon_qqplots.xlsx')
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 1. Guardar primero la pestaña de Resumen
        summary_df.to_excel(writer, sheet_name="Resumen_Efectos", index=False)
        
        # 2. Guardar el resto de pestañas de datasets
        for dataset in datasets:
            for scope in scopes:
                slice_df = results_df[(results_df['Dataset'] == dataset) & (results_df['Caso'] == scope)].copy()
                if not slice_df.empty:
                    sheet_name = f"{dataset[:20]} - {scope}"
                    slice_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"-> ¡Éxito! Excel generado en: '{excel_path}' (Incluye pestaña 'Resumen_Efectos')")
    print(f"-> Matrices Q-Q Plots guardadas en: '{qq_dir}'")
    return results_df

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    hist_file = os.path.join(script_dir, 'tfg_results.xlsx') 
    new_file = os.path.join(script_dir, 'tfg_results_mymodels.xlsx') 
    output_directory = os.path.join(script_dir, 'exported_results_mymodels', 'statistics')
    
    if not os.path.exists(hist_file) or not os.path.exists(new_file):
        print("Error: No se encontraron los archivos Excel.")
        return

    df = load_and_combine_data(hist_file, new_file)
    run_tests_and_qqplots(df, output_directory)

if __name__ == "__main__":
    main()