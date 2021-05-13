import transformers
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
model = AutoModel.from_pretrained("avichr/heBERT", output_hidden_states = True)

df = pd.read_csv('dataset_bert.txt', index_col=0)

embedded_sentences = []
for sent in df['tokenized_text_sent']:
    marked_sent = "[CLS] " + sent + " [SEP]"
    indexed_tokens = tokenizer(marked_sent)['input_ids']
    segments_ids = tokenizer(marked_sent)['attention_mask']
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
    token_vecs = hidden_states[-2][0]
    sentence_embedding = torch.mean(token_vecs, dim=0)
    embedded_sentences.append(sentence_embedding)

print(embedded_sentences)
embedded_df = pd.DataFrame(embedded_sentences)
embedded_df.to_csv('embedded_df.txt')