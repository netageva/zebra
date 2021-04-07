from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained, SUPPORTED_MLMS
from transformers import AutoTokenizer, AutoModel
import mxnet as mx
import pandas as pd
import gluonnlp
import streamlit as st


ctxs = [mx.cpu()]

'''filename = 'vocab.txt'
text_data = []
with open(filename, encoding="utf8") as fh:
    for line in fh:
        word = line.strip()
        text_data.append(word)

counter = gluonnlp.data.count_tokens(text_data)
vocab = gluonnlp.vocab.BERTVocab(counter)
tokenizer = AutoTokenizer.from_pretrained("avichr/heBERT")
model = AutoModel.from_pretrained("avichr/heBERT", output_hidden_states = True)'''

model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-multi-cased')
scorer = MLMScorer(model, vocab, tokenizer, ctxs)
print(scorer.score_sentences(["הי קוראים לי נטע"]))
print(scorer.score_sentences(["הי קוראים לי נטע"], per_token=True))

df = pd.read_csv('dataset_bert.txt')
df['score'] = 0

'''for i,row in df.iterrows():
    sent = [row['tokenized_text_sent']]
    df.loc[i,'score'] = scorer.score_sentences(sent)
df.to_csv('mlm_scoring_result.txt')'''


