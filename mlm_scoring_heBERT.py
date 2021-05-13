from transformers import AutoTokenizer, BertForMaskedLM
from torch.nn import functional as F
import numpy as np
import config as CFG

class BertMlmScoring():
    def __init__(self):
        super(BertMlmScoring, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(CFG.HE_BERT)
        self.model = BertForMaskedLM.from_pretrained(CFG.HE_BERT, output_hidden_states=True, return_dict=True)
        self.sent_score = 0

    def calc_score_for_sent(self, text):
        token_text = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
        self.sent_score = 0
        for i in range(len(token_text[CFG.IDS_KEY][0])):
            token_text = self.tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt")
            token_text[CFG.IDS_KEY][0][i] = self.tokenizer.mask_token_id
            word_index = token_text[CFG.IDS_KEY][0][i]
            output = self.model(**token_text)
            logits = output.logits
            softmax = F.softmax(logits, dim=-1)
            mask_word = softmax[0, i, :]
            word_score = np.log((mask_word[word_index]).detach().numpy())
            self.sent_score += word_score
            self.sent_score = self.sent_score/len(token_text[CFG.IDS_KEY][0])
        return self.sent_score



























'''
#prediction_logits
#take bert embedding layer and multiple it with y(dim=3)
m = nn.Softmax(dim=3)
proba = m(y)
(1,512,50000) - log on dim 3
print(proba.shape)
proba = proba.detach().numpy()
log_proba = np.log(proba)
sum_log = np.sum(log_proba)
print('score:',sum_log)'''
