# mlm scoring for heBERT

![Image](https://github.com/netageva/zebra/blob/master/mlm-scoring.png)

This model uses the method suggested by the autors of [Masked Language Model Scoring]. 
The goal is to evaluate MLMs via their pseudo-log-likelihood scores (PLLs), which are computed by masking tokens one by one.


The code here implement the mlm-scoring method using [heBERT]. 

### py files:
- mlm_scroing_heBERT.py: Includes the class BertMlmScoring with the function calc_score_for_sent that calculate the score per sentence input. The class already contains the model and tokenizer
- mlm_scoring_app.py: Given an input (CSV: file with one column of samples), this code will call the BertMlmScoring class and insert the results to dataframe. 

### how to use:
- Insert the path to samples file to CSV variable in the config file.
- In the terminal: 
    ```
    streamlit run mlm_scoring_app.py
    ```
- The result will appear in a new tab at your browser.


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

[Masked Language Model Scoring]: <https://www.aclweb.org/anthology/2020.acl-main.240.pdf>
[heBERT]: <https://huggingface.co/avichr/heBERT>










