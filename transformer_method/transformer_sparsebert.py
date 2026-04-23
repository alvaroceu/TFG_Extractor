import torch
import math
import torch.nn.functional as F
from transformers.models.distilbert.modeling_distilbert import MultiHeadSelfAttention
from transformers import pipeline
from core.extractor_base import ExtractorBase
from core.preprocessing import *

# =====================================================================
# 1. ATENCIÓN POR BLOQUES AISLADOS (Máxima Velocidad, Peor Calidad)
# =====================================================================
class RealBlockSparseAttention(MultiHeadSelfAttention):
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

        q_blocks = q_blocks / math.sqrt(dim_per_head)
        scores_blocks = torch.matmul(q_blocks, k_blocks.transpose(-1, -2)) 

        weights_blocks = F.softmax(scores_blocks, dim=-1)
        weights_blocks = self.dropout(weights_blocks)

        context_blocks = torch.matmul(weights_blocks, v_blocks)
        context = context_blocks.view(bs, self.n_heads, new_len, dim_per_head)
        
        if pad_len > 0:
            context = context[:, :, :q_length, :]

        context = unshape(context)
        context = self.out_lin(context)

        return (context, weights_blocks) if output_attentions else (context,)

# =====================================================================
# 2. BIG BIRD LITE (Velocidad Real + Puente Global CLS)
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

        # --- A) Calcular la atención global del token [CLS] hacia todo el texto ---
        q_cls = q[:, :, 0:1, :] # Extraemos solo el primer token
        scores_cls = torch.matmul(q_cls / math.sqrt(dim_per_head), k.transpose(-1, -2))
        weights_cls = self.dropout(F.softmax(scores_cls, dim=-1))
        context_cls = torch.matmul(weights_cls, v) # (bs, heads, 1, dim_per_head)

        # --- B) Calcular los bloques, pero forzando a que todos miren también al [CLS] ---
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

        # Expandimos el CLS para poder concatenarlo a cada uno de los bloques
        k_cls_expanded = k[:, :, 0:1, :].unsqueeze(2).expand(bs, self.n_heads, num_blocks, 1, dim_per_head)
        v_cls_expanded = v[:, :, 0:1, :].unsqueeze(2).expand(bs, self.n_heads, num_blocks, 1, dim_per_head)

        # Unimos el CLS al inicio de cada bloque (El tamaño de la key pasa a ser B + 1)
        k_combined = torch.cat([k_cls_expanded, k_blocks], dim=3)
        v_combined = torch.cat([v_cls_expanded, v_blocks], dim=3)

        q_blocks = q_blocks / math.sqrt(dim_per_head)
        scores_blocks = torch.matmul(q_blocks, k_combined.transpose(-1, -2)) 
        
        weights_blocks = self.dropout(F.softmax(scores_blocks, dim=-1))
        context_blocks = torch.matmul(weights_blocks, v_combined)

        context = context_blocks.view(bs, self.n_heads, new_len, dim_per_head)
        if pad_len > 0:
            context = context[:, :, :q_length, :]

        # --- C) Sobrescribir el resultado del CLS real en la posición 0 ---
        context[:, :, 0:1, :] = context_cls

        context = unshape(context)
        context = self.out_lin(context)

        return (context, weights_blocks) if output_attentions else (context,)

# =====================================================================
# ENRUTADOR PRINCIPAL
# =====================================================================
def inject_attention(model, attention_type="block", block_size=64, num_random_tokens=3):
    """
    Inyecta la capa de atención seleccionada en DistilBERT.
    Opciones de attention_type: 'block', 'bigbird_lite', 'bigbird_simulated'
    """
    print(f"Inyectando arquitectura: {attention_type.upper()}")
    
    for i, layer in enumerate(model.distilbert.transformer.layer):
        old_attention = layer.attention
        
        if attention_type == "block":
            new_attention = RealBlockSparseAttention(model.config, block_size=block_size)
        elif attention_type == "bigbird_lite":
            new_attention = BigBirdLiteAttention(model.config, block_size=block_size)
        else:
            raise ValueError(f"Tipo de atención desconocido: {attention_type}")
        
        # Transferencia de conocimiento
        new_attention.q_lin.load_state_dict(old_attention.q_lin.state_dict())
        new_attention.k_lin.load_state_dict(old_attention.k_lin.state_dict())
        new_attention.v_lin.load_state_dict(old_attention.v_lin.state_dict())
        new_attention.out_lin.load_state_dict(old_attention.out_lin.state_dict())
        
        new_attention.to(old_attention.q_lin.weight.device)
        layer.attention = new_attention
        
    return model

# =====================================================================
# CLASE EXTRACTORA
# =====================================================================
class TransformerSparseBertExtractor(ExtractorBase):
    
    # Hemos añadido todos los parámetros al constructor
    def __init__(self, attention_type="bigbird_lite", block_size=64, num_random_tokens=3):
        self.sparse_model = pipeline('question-answering', model='twmkn9/distilbert-base-uncased-squad2', tokenizer='twmkn9/distilbert-base-uncased-squad2', device=0)
        
        # Inyectamos el modelo seleccionado
        self.sparse_model.model = inject_attention(
            self.sparse_model.model, 
            attention_type=attention_type, 
            block_size=block_size,
            num_random_tokens=num_random_tokens
        )

    def extract(self, text: str, questions: str):
        results = {}
        parsed_questions = parse_questions_embeddings(questions)

        for key, question in parsed_questions.items():
            result = self.sparse_model(question=question, context=text, handle_impossible_answer=True)

            if result['answer'].strip() == "":
                best_answer = "A possible valid answer wasn't found"
            else:
                best_answer = result['answer']

            results[key] = best_answer
            print(f"{key}: score={result['score']:.3f} | answer={best_answer}")

        return results