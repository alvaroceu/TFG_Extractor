import time
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from transformers import pipeline
from transformers.models.bert.modeling_bert import BertSelfAttention
from core.extractor_base import ExtractorBase
from core.preprocessing import *

# =Big bird based approach
class BertBigBirdLiteSelfAttention(BertSelfAttention):
    def __init__(self, config, position_embedding_type=None, block_size=64):
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.block_size = block_size

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        bs, n_heads, q_length, head_dim = query_layer.size()

        # [cls] token
        query_cls = query_layer[:, :, 0:1, :]
        attention_scores_cls = torch.matmul(query_cls, key_layer.transpose(-1, -2)) / math.sqrt(head_dim)
        
        if attention_mask is not None:
            attention_scores_cls = attention_scores_cls + attention_mask

        attention_probs_cls = nn.functional.softmax(attention_scores_cls, dim=-1)
        attention_probs_cls = self.dropout(attention_probs_cls)
        context_cls = torch.matmul(attention_probs_cls, value_layer)

        # Attention block
        pad_len = (self.block_size - q_length % self.block_size) % self.block_size
        
        if pad_len > 0:
            query_layer = F.pad(query_layer, (0, 0, 0, pad_len))
            key_layer = F.pad(key_layer, (0, 0, 0, pad_len))
            value_layer = F.pad(value_layer, (0, 0, 0, pad_len))

        new_len = query_layer.size(2)
        num_blocks = new_len // self.block_size

        q_blocks = query_layer.view(bs, n_heads, num_blocks, self.block_size, head_dim)
        k_blocks = key_layer.view(bs, n_heads, num_blocks, self.block_size, head_dim)
        v_blocks = value_layer.view(bs, n_heads, num_blocks, self.block_size, head_dim)

        k_cls_expanded = key_layer[:, :, 0:1, :].unsqueeze(2).expand(bs, n_heads, num_blocks, 1, head_dim)
        v_cls_expanded = value_layer[:, :, 0:1, :].unsqueeze(2).expand(bs, n_heads, num_blocks, 1, head_dim)

        # CLS + block
        k_combined = torch.cat([k_cls_expanded, k_blocks], dim=3)
        v_combined = torch.cat([v_cls_expanded, v_blocks], dim=3)

        attention_scores_blocks = torch.matmul(q_blocks, k_combined.transpose(-1, -2)) / math.sqrt(head_dim)

        if attention_mask is not None:
            cls_mask = attention_mask[:, :, :, 0:1] 
            
            if pad_len > 0:
                mask_padded = F.pad(attention_mask, (0, pad_len), value=attention_mask.min().item())
            else:
                mask_padded = attention_mask
                
            mask_blocks = mask_padded.view(bs, 1, num_blocks, 1, self.block_size)
            cls_mask_expanded = cls_mask.unsqueeze(2).expand(bs, 1, num_blocks, 1, 1)
            
            k_mask_combined = torch.cat([cls_mask_expanded, mask_blocks], dim=-1) 
            
            attention_scores_blocks = attention_scores_blocks + k_mask_combined

        attention_probs_blocks = nn.functional.softmax(attention_scores_blocks, dim=-1)
        attention_probs_blocks = self.dropout(attention_probs_blocks)
        
        context_blocks = torch.matmul(attention_probs_blocks, v_combined)
        context_layer_blocks = context_blocks.view(bs, n_heads, new_len, head_dim)

        if pad_len > 0:
            context_layer_blocks = context_layer_blocks[:, :, :q_length, :]

        context_layer_blocks[:, :, 0:1, :] = context_cls

        context_layer = context_layer_blocks.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs_blocks) if output_attentions else (context_layer,)
        return outputs

def inject_bert_attention(model, block_size=64):
    """
    Change attention layer in DistilBERT. Insert our custom implementation.
    """
    
    for i, layer in enumerate(model.bert.encoder.layer):

        old_attention = layer.attention.self 
        
        new_attention = BertBigBirdLiteSelfAttention(model.config, block_size=block_size)

        # Transfer knowledge
        new_attention.query.load_state_dict(old_attention.query.state_dict())
        new_attention.key.load_state_dict(old_attention.key.state_dict())
        new_attention.value.load_state_dict(old_attention.value.state_dict())
        
        new_attention.to(device=old_attention.query.weight.device, dtype=old_attention.query.weight.dtype)
        
        layer.attention.self = new_attention
        
    return model

# New Sparse class
class TransformerSparseBertLargeExtractor(ExtractorBase):

    def __init__(self, block_size=64):
        self.sparsemodel = pipeline(
            'question-answering', 
            model='deepset/bert-large-uncased-whole-word-masking-squad2', 
            tokenizer='deepset/bert-large-uncased-whole-word-masking-squad2', 
            torch_dtype=torch.float16, 
            device=0
        )
        
        self.sparsemodel.model = inject_bert_attention(
            self.sparsemodel.model, 
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
            
            result = self.sparsemodel(question=question,context=text,handle_impossible_answer=True)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times[key] = time.perf_counter() - start_time

            if result['answer'].strip() == "":
                best_answer = "A possible valid answer wasn't found"
            else:
                best_answer = result['answer']

            results[key] = best_answer

        return results, times