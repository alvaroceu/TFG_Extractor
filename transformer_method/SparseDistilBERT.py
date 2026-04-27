import time
import torch
import math
import torch.nn.functional as F
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention
from transformers import pipeline
from core.extractor_base import ExtractorBase
from core.preprocessing import *

# =====================================================================
# 1. BIG BIRD LITE (Velocidad Real + Puente Global CLS)
# =====================================================================
class BigBirdLiteAttention(MultiHeadSelfAttention):
    def __init__(self, config, block_size=64, **kwargs):
        super().__init__(config)
        self.block_size = block_size

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        bs, q_length, dim = query.size()
        dim_per_head = self.dim // self.n_heads

        def shape(x): return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)
        def unshape(x): return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))
        k = shape(self.k_lin(key))
        v = shape(self.v_lin(value))
        v_orig = v

        # --- A) Atención global del token [CLS] ---
        q_cls = q[:, :, 0:1, :] 
        scores_cls = torch.matmul(q_cls / math.sqrt(dim_per_head), k.transpose(-1, -2))

        # --- B) Calcular los bloques ---
        pad_len = (self.block_size - q_length % self.block_size) % self.block_size
        if pad_len > 0:
            q = F.pad(q, (0, 0, 0, pad_len))
            k = F.pad(k, (0, 0, 0, pad_len))
            v = F.pad(v, (0, 0, 0, pad_len))
        
        new_len = q.size(2)
        num_blocks = new_len // self.block_size

        q_blocks = q.view(bs, self.n_heads, num_blocks, self.block_size, dim_per_head)
        k_blocks = k.view(bs, self.n_heads, num_blocks, self.block_size, dim_per_head)
        v_blocks = v.view(bs, self.n_heads, num_blocks, self.block_size, dim_per_head)

        k_cls_expanded = k[:, :, 0:1, :].unsqueeze(2).expand(bs, self.n_heads, num_blocks, 1, dim_per_head)
        v_cls_expanded = v[:, :, 0:1, :].unsqueeze(2).expand(bs, self.n_heads, num_blocks, 1, dim_per_head)

        k_combined = torch.cat([k_cls_expanded, k_blocks], dim=3)
        v_combined = torch.cat([v_cls_expanded, v_blocks], dim=3)

        q_blocks = q_blocks / math.sqrt(dim_per_head)
        scores_blocks = torch.matmul(q_blocks, k_combined.transpose(-1, -2)) 

        # APLICAR LA MÁSCARA ANTES DEL SOFTMAX
        if mask is not None:
            # En DistilBERT, mask tiene 1 para tokens reales y 0 para padding
            
            # 1. Enmascarar el CLS
            mask_cls_bool = (mask == 0).view(bs, 1, 1, q_length)
            scores_cls = scores_cls.masked_fill(mask_cls_bool, torch.tensor(torch.finfo(scores_cls.dtype).min))
            
            # 2. Enmascarar los bloques
            # Pad a la máscara igual que a los tensores
            mask_padded = F.pad(mask, (0, pad_len), value=0)
            mask_blocks = mask_padded.view(bs, num_blocks, self.block_size)
            
            # Máscara para el CLS que se une a los bloques
            cls_mask = mask[:, 0:1].unsqueeze(1).expand(bs, num_blocks, 1)
            
            # Máscara combinada (CLS + Bloque)
            k_mask_combined = torch.cat([cls_mask, mask_blocks], dim=2)
            k_mask_bool = (k_mask_combined == 0).view(bs, 1, num_blocks, 1, 1 + self.block_size)
            
            scores_blocks = scores_blocks.masked_fill(k_mask_bool, torch.tensor(torch.finfo(scores_blocks.dtype).min))
        # ==========================================================

        weights_cls = self.dropout(F.softmax(scores_cls, dim=-1))
        context_cls = torch.matmul(weights_cls, v_orig)

        weights_blocks = self.dropout(F.softmax(scores_blocks, dim=-1))
        context_blocks = torch.matmul(weights_blocks, v_combined)

        context = context_blocks.view(bs, self.n_heads, new_len, dim_per_head)
        if pad_len > 0:
            context = context[:, :, :q_length, :]

        context[:, :, 0:1, :] = context_cls

        context = unshape(context)
        context = self.out_lin(context)

        return (context, weights_blocks) if output_attentions else (context,)

# =====================================================================
# ENRUTADOR PRINCIPAL
# =====================================================================
def inject_attention(model, block_size=64):
    """
    Inyecta la capa de atención seleccionada en DistilBERT.
    """
    
    for i, layer in enumerate(model.distilbert.transformer.layer):
        old_attention = layer.attention
        
        new_attention = BigBirdLiteAttention(model.config, block_size=block_size)

        # Transferencia de conocimiento
        new_attention.q_lin.load_state_dict(old_attention.q_lin.state_dict())
        new_attention.k_lin.load_state_dict(old_attention.k_lin.state_dict())
        new_attention.v_lin.load_state_dict(old_attention.v_lin.state_dict())
        new_attention.out_lin.load_state_dict(old_attention.out_lin.state_dict())
        
        new_attention.to(device=old_attention.q_lin.weight.device, dtype=old_attention.q_lin.weight.dtype)
        layer.attention = new_attention
        
    return model

# =====================================================================
# CLASE EXTRACTORA
# =====================================================================
class TransformerSparseDistilBertExtractor(ExtractorBase):
    
    # Hemos añadido todos los parámetros al constructor
    def __init__(self, block_size=64):
        self.sparse_model = pipeline('question-answering', model='twmkn9/distilbert-base-uncased-squad2', tokenizer='twmkn9/distilbert-base-uncased-squad2', torch_dtype=torch.float16, device=0)
        
        # Inyectamos el modelo seleccionado
        self.sparse_model.model = inject_attention(
            self.sparse_model.model, 
            block_size=block_size
        )

    def extract(self, text: str, questions: str):
        results = {}
        times = {}

        parsed_questions = parse_questions_embeddings(questions)

        for key, question in parsed_questions.items():

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            result = self.sparse_model(question=question, context=text, handle_impossible_answer=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times[key] = time.perf_counter() - start_time

            if result['answer'].strip() == "":
                best_answer = "A possible valid answer wasn't found"
            else:
                best_answer = result['answer']

            results[key] = best_answer

        return results, times