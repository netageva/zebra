import streamlit as st
from mlm_scoring_heBERT import *
import pandas as pd
import config as CFG

df = pd.read_csv(CFG.CSV, delimiter ="\n", header=None, names=[CFG.DF_COL])
all_sent = list(df[CFG.DF_COL])
bertmlm = BertMlmScoring()

scores = {}
for sent in range(len(all_sent)):
    score = bertmlm.calc_score_for_sent((all_sent[sent])).item()
    scores[sent] = {'input': all_sent[sent],
                    'score': score}
    score = 0

dataframe = pd.DataFrame.from_dict(data=scores, columns=['input','score'], orient='index')
dataframe = dataframe.sort_values('score')
st.header("mlm scores for reports")
st.table(dataframe)

