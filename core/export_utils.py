import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment

def export_predictions_to_excel(all_predictions, all_references, excel_name="resultados_textos.xlsx"):
    """
    Exporta las respuestas de texto. 
    Filas: Cada una de las preguntas de todos los contextos.
    Columnas: Ground Truth y los Modelos.
    """
    # Construimos el diccionario base para el DataFrame
    data = {
        "Ground Truth": all_references
    }
    # Añadimos las listas de predicciones de cada modelo
    data.update(all_predictions)

    # Convertimos a DataFrame
    df = pd.DataFrame(data)
    
    # Exportamos a Excel
    df.to_excel(excel_name, index=False)
    
    # Aplicar tu formato bonito de autoajuste
    _format_excel(excel_name)

def export_metrics_to_excel(all_metrics, excel_name="resultados_metricas.xlsx"):
    """
    Exporta las notas matemáticas.
    Filas: Modelos (BoW, BERT, etc.)
    Columnas: Las métricas (Exact Match, Falsos Positivos, etc.)
    """
    # Convertimos el diccionario de métricas a DataFrame (orient='index' pone las claves principales como filas)
    df = pd.DataFrame.from_dict(all_metrics, orient='index')
    
    # Reseteamos el índice para que los nombres de los modelos sean una columna
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'Modelo'}, inplace=True)
    
    df.to_excel(excel_name, index=False)
    _format_excel(excel_name)

def _format_excel(excel_name):
    """Función auxiliar con tu código original para poner bonito el Excel."""
    wb = load_workbook(excel_name)
    ws = wb.active

    for col_idx, col in enumerate(ws.iter_cols(min_row=1, max_row=ws.max_row, max_col=ws.max_column), start=1):
        max_length = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        adjusted_width = min(max_length + 2, 40)
        ws.column_dimensions[get_column_letter(col_idx)].width = adjusted_width

        for cell in col:
            cell.alignment = Alignment(wrap_text=True, vertical='top')

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        ws.row_dimensions[row[0].row].height = 60

    wb.save(excel_name)
