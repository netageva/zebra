from tokenizer import *
import streamlit as st
import pandas as pd
import numpy as np

SEED = 10

all_reports = pd.read_csv('sample_clalit_reports - clalit_reports_5_precent.csv')
ind = np.random.randint(all_reports.shape[0])
report = all_reports['text'][ind]

tokens = ReportTokenizer(report).create_tokens_sentences()
original = [sent.strip() + '.' for sent in report.strip().split('.') if sent.strip() != '']

table = {'tokens':tokens, 'original':original}
dataframe = pd.DataFrame(data=table)

st.header("Tokenized Report vs Original")
st.table(dataframe)
#st.write(ReportTokenizer(report).create_tokens_sentences())



