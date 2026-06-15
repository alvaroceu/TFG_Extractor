import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import warnings

warnings.filterwarnings("ignore")

def main():
    print("Generating heatmaps")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # load data
    hist_file = os.path.join(script_dir, 'tfg_results.xlsx')
    new_file = os.path.join(script_dir, 'tfg_results_mymodels.xlsx')
    cong_file = os.path.join(script_dir, 'tfg_results_mymodels_congelado.xlsx')
    
    output_dir = os.path.join(script_dir, 'exported_results_mymodels')
    os.makedirs(output_dir, exist_ok=True)
    
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
    datasets = ['SQuAD 2.0', 'NewsQA', 'Natural Questions', 'TriviaQA']
    
    df_all = pd.concat([
        df_hist[df_hist['Model'].isin(model_list)],
        df_new[df_new['Model'].isin(model_list)],
        df_cong[df_cong['Model'].isin(['BiLBERT-Distil (Congelado)', 'BiLBERT-Large (Congelado)'])]
    ], ignore_index=True)
    
    if df_all['ExecTime'].dtype == object:
        df_all['ExecTime'] = df_all['ExecTime'].astype(str).str.replace(',', '.')
    df_all['ExecTime'] = pd.to_numeric(df_all['ExecTime'], errors='coerce')
    
    mean_times_df = pd.DataFrame(index=model_list, columns=datasets)
    r_wilcoxon_df = pd.DataFrame(index=model_list, columns=datasets)
    
    for dataset in datasets:
        df_ds = df_all[df_all['Dataset'] == dataset]
        
        for m1 in model_list:
            df_m1 = df_ds[df_ds['Model'] == m1][['Question ID', 'ExecTime']].dropna()
            
            # mean time
            if len(df_m1) > 0:
                mean_times_df.loc[m1, dataset] = df_m1['ExecTime'].mean()
            else:
                mean_times_df.loc[m1, dataset] = np.nan
                
            # Wilcoxon r effect
            r_values = []
            for m2 in model_list:
                if m1 != m2:
                    df_m2 = df_ds[df_ds['Model'] == m2][['Question ID', 'ExecTime']].dropna()
                    merged = pd.merge(df_m1, df_m2, on='Question ID')
                    
                    if len(merged) > 10:
                        data_a = merged['ExecTime_x'].values
                        data_b = merged['ExecTime_y'].values
                        try:
                            res = stats.wilcoxon(data_a, data_b, alternative='two-sided', method='approx')
                            effect_size_r = abs(res.zstatistic) / np.sqrt(len(data_a))
                            r_values.append(effect_size_r)
                        except ValueError:
                            pass
                            
            if r_values:
                r_wilcoxon_df.loc[m1, dataset] = np.mean(r_values)
            else:
                r_wilcoxon_df.loc[m1, dataset] = np.nan

    mean_times_df = mean_times_df.astype(float)
    r_wilcoxon_df = r_wilcoxon_df.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Análisis de Tiempos de Ejecución: Medias vs. Magnitud del Efecto Global', 
                 fontsize=22, fontweight='bold', y=1.02)
    
    # Heatmap 1: Mean times
    sns.heatmap(mean_times_df, ax=axes[0], annot=True, fmt=".4f", 
                cmap="RdYlGn_r", cbar_kws={'label': ''}, linewidths=.5,
                annot_kws={"size": 14, "weight": "bold"})
    
    axes[0].set_title('Tiempo Medio de Ejecución (s)', fontsize=18, pad=15)
    axes[0].set_ylabel('')
    axes[0].set_xlabel('')
    axes[0].tick_params(axis='y', rotation=0, labelsize=14) 
    axes[0].tick_params(axis='x', labelsize=14)
    
    # Heatmap 2: Wilcoxon r effect
    sns.heatmap(r_wilcoxon_df, ax=axes[1], annot=True, fmt=".3f", 
                cmap="Oranges", cbar_kws={'label': ''}, linewidths=.5,
                annot_kws={"size": 14, "weight": "bold"})
                
    axes[1].set_title('Efecto Medio "r" (Diferencia vs. el resto de modelos)', fontsize=18, pad=15)
    axes[1].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].tick_params(axis='y', left=False, labelleft=False) 
    axes[1].tick_params(axis='x', labelsize=14)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1)
    
    filename = os.path.join(output_dir, 'heatmap_comparativa_tiempos.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Heatmap saved in '{filename}'.")

if __name__ == "__main__":
    main()