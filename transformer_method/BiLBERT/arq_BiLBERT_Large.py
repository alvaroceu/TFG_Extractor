import torch
import torch.nn as nn
from transformers import BertModel

class BiLBERTLarge(nn.Module):
    def __init__(self, model_name='deepset/bert-large-uncased-whole-word-masking-squad2', hidden_size=256):

        # Download the BERTLarge model
        super(BiLBERTLarge, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        self.lstm = nn.LSTM(
            input_size=1024, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True, 
            bidirectional=True
        )
        
        # Final clasfier layer
        self.qa_outputs = nn.Linear(1024 + (hidden_size * 2), 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        
        # Through BERTLarge
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = bert_output[0]
        sequence_output_dropped = self.dropout(sequence_output)

        # Through BiLSTM
        lstm_output, _ = self.lstm(sequence_output_dropped) 
        combined_output = torch.cat((sequence_output, lstm_output), dim=-1)
        combined_output = self.dropout(combined_output)

        # Through final linear layer
        logits = self.qa_outputs(combined_output) 
        
        # Results: Start and End logits
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1) 
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits