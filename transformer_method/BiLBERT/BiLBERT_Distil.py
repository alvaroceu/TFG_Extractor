import time
import torch
from transformers import AutoTokenizer
from typing import Tuple, Dict

from core.extractor_base import ExtractorBase
from core.preprocessing import parse_questions_embeddings

# Asegúrate de que este import coincida con la ruta donde guardaste tu clase
from .arq_BiLBERT_Distil import BiLBERTDistil

class TransformerBiLBERTDistilExtractor(ExtractorBase):
    def __init__(self, model_weights_path="trained_models/bilbert_distil_qa_weights.pth"):
        print("📥 Cargando Tokenizador y arquitectura BiLBERT...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('twmkn9/distilbert-base-uncased-squad2')
        
        # 1. Instanciar el modelo "vacío" (cuerpo + LSTM)
        self.model = BiLBERTDistil()
        
        # 2. Inyectar tu cerebro entrenado
        print(f"🧠 Inyectando conocimiento desde: {model_weights_path}")
        self.model.load_state_dict(torch.load(model_weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.half()
        self.model.eval() # Modo inferencia (vital para que no intente seguir aprendiendo)

    def extract(self, text: str, questions: str) -> Tuple[Dict[str, str], Dict[str, float]]:
        results = {}
        times = {}
        parsed_questions = parse_questions_embeddings(questions)

        with torch.no_grad(): # Apagamos gradientes para máxima velocidad
            for key, question in parsed_questions.items():
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # A. Tokenización manual (Lo que antes hacía la pipeline por ti)
                inputs = self.tokenizer(
                    question, 
                    text, 
                    return_tensors="pt", 
                    truncation="only_second", # Si es muy largo, cortamos el texto, no la pregunta
                    max_length=512,
                    stride=128,
                    return_overflowing_tokens=True,
                    padding="max_length"
                )

                input_ids_all = inputs["input_ids"].to(self.device)
                attention_mask_all = inputs["attention_mask"].to(self.device)
                num_chunks = input_ids_all.shape[0] # Ej: Si el texto es largo, num_chunks será 10

                best_score = float('-inf')
                best_answer = "A possible valid answer wasn't found"

                # NUEVO: Bucle que evalúa cada trozo en la gráfica
                for i in range(num_chunks):
                    input_ids = input_ids_all[i].unsqueeze(0)
                    attention_mask = attention_mask_all[i].unsqueeze(0)

                    start_logits, end_logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )

                    start_index = torch.argmax(start_logits)
                    end_index = torch.argmax(end_logits)

                    if end_index >= start_index and start_index > 0:
                        # NUEVO: Calculamos la nota (score) matemática de esta respuesta
                        score = start_logits[0, start_index].item() + end_logits[0, end_index].item()

                        # NUEVO: Si la nota es mejor que la anterior, actualizamos la mejor respuesta
                        if score > best_score:
                            best_score = score
                            predict_answer_tokens = input_ids[0][start_index : end_index + 1]
                            best_answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

                if best_answer.strip() == "":
                    best_answer = "A possible valid answer wasn't found"

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                times[key] = time.perf_counter() - start_time
                
                results[key] = best_answer

        return results, times