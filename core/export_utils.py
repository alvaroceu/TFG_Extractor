import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment

def export_results_to_excel(questions, results_dict, excel_name):
    """Exports an Excel with the generated answers to questions for different articles."""

    # Get de key for each question (instead of the hole question)
    question_list = []
    for line in questions.strip().splitlines():
        if ':' in line:
            key, _ = line.strip().split(':', 1)
            question_list.append(key.strip())

    # Build Excel
    rows = []
    for paper_name, result in results_dict.items():
        row = {'Paper name': paper_name}
        for key in question_list:
            row[key] = result.get(key, "") 
        rows.append(row)

    # Create DataFrame and export
    df = pd.DataFrame(rows)
    if not excel_name.endswith('.xlsx'):
        excel_name += '.xlsx'
    df.to_excel(excel_name, index=False)

    # Format
    wb = load_workbook(excel_name)
    ws = wb.active

    # Adjust columns and rows, apply text adjustment
    for col_idx, col in enumerate(ws.iter_cols(min_row=1, max_row=ws.max_row, max_col=ws.max_column), start=1):
        max_length = max(len(str(cell.value)) if cell.value else 0 for cell in col)
        adjusted_width = min(max_length + 2, 40)  # Limits max width
        ws.column_dimensions[get_column_letter(col_idx)].width = adjusted_width

        for cell in col:
            cell.alignment = Alignment(wrap_text=True, vertical='top')

    # Adjust row height
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        ws.row_dimensions[row[0].row].height = 60

    wb.save(excel_name)
