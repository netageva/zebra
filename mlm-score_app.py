import pandas as pd
import streamlit as st

df_scores = pd.read_csv('mlm_scoring_result.txt', index_col=0)
df_scores = df_scores.sort_values(by='score')
df_scores.drop_duplicates(subset ="tokenized_text_sent", inplace = True)

sent = df_scores['tokenized_text_sent']
score = df_scores['score']

table = {'sentence':sent, 'score':score}
dataframe = pd.DataFrame(data=table)

st.header("mlm scores for multiBERT")
st.table(dataframe)