import torch
import torch.nn as nn
from transformers import DistilBertModel

class BiLBERTDistil(nn.Module):
    def __init__(self, model_name='twmkn9/distilbert-base-uncased-squad2', hidden_size=256):
        # 1. Inicializamos la clase padre (Obligatorio en PyTorch)
        super(BiLBERTDistil, self).__init__()
        
        # 2. EL CUERPO: Cargamos el modelo base de DistilBERT. 
        # Fíjate que usamos 'DistilBertModel' y no 'ForQuestionAnswering'. 
        # Esto nos da el Transformer puro, sin la "cabeza" tonta original.
        self.distilbert = DistilBertModel.from_pretrained(model_name)
        
        # 3. CONGELACIÓN: Le decimos a PyTorch que no modifique los pesos de DistilBERT.
        # ¿Por qué? Porque ya sabe inglés. Solo queremos enseñar a la nueva cabeza LSTM.
        for param in self.distilbert.parameters():
            param.requires_grad = False
            
        # 4. EL PUENTE: La red Bi-LSTM.
        # Input: 768 (Porque DistilBERT escupe vectores de tamaño 768 por cada palabra)
        # Output: 256 * 2 (Porque es bidireccional, lee hacia adelante y hacia atrás)
        self.lstm = nn.LSTM(
            input_size=768, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # 5. EL CLASIFICADOR FINAL: 
        # Coge los 512 números que escupe el LSTM y los aplasta a solo 2 números por palabra:
        # Probabilidad de [Inicio] y Probabilidad de [Fin]
        self.qa_outputs = nn.Linear(hidden_size * 2, 2)

    def forward(self, input_ids, attention_mask):
        # A) El texto (números) pasa por el Transformer
        distilbert_output = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        
        # B) Extraemos los "estados ocultos". Es una matriz con tamaño: (Batch, Num_Palabras, 768)
        sequence_output = distilbert_output[0]
        
        # C) Pasamos esa matriz por nuestra nueva capa Bi-LSTM
        lstm_output, _ = self.lstm(sequence_output) # Se convierte en (Batch, Num_Palabras, 512)
        
        # D) Pasamos el resultado por la capa lineal final
        logits = self.qa_outputs(lstm_output) # Se convierte en (Batch, Num_Palabras, 2)
        
        # E) Separamos los 2 números finales. Uno es para el inicio, otro para el fin.
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) # Quitamos dimensiones vacías
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits