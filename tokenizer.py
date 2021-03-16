class Tokenizer:

    def __init__(self, sentence):
        import hebrew_tokenizer as ht

        self.sentence = sentence
        self.tokens = ht.tokenize(self.sentence)

        self.result = []
        for grp, token, token_num, (start_index, end_index) in self.tokens:
            self.result.append((grp, token))

    def as_list(self):
        return [word[1] for word in self.result]

    def as_list_no_punc(self):
        return [word[1] for word in self.result if word[0] != 'PUNCTUATION']

    def pos_and_token(self):
        return self.result

    def pos_and_token_no_punc(self):
        return [(word[0], word[1]) for word in self.result if word[0] != 'PUNCTUATION']

    def customized(self):
        to_return = []
        for grp,token in self.result:
            if grp == 'PUNCTUATION':
                if token in ':,()':
                    to_return.append(token)
            else:
                to_return.append(token)

        return to_return


class ReportTokenizer:

    def __init__(self, report):
        self.report = report


    def create_tokens_sentences(self):

        sents = [sent.strip() + '.' for sent in self.report.strip().split('\n') if sent.strip() != '']
        sents_token = [Tokenizer(sent).customized() for sent in sents]
        sents_join = [' '.join(sent) for sent in sents_token]

        return sents_join

    def create_tokens_split_sentences(self):

        sents = [sent.strip() + '.' for sent in self.report.strip().split('\n') if sent.strip() != '']
        sents_token = [Tokenizer(sent).customized() for sent in sents]

        return sents_token

    def save_to_file(self, file_name):
        token_sents = self.create_tokens_sentences()
        with open(str(file_name) +'.txt', 'w', encoding="utf-8") as out:
            for line in token_sents:
                out.write(line + '\n')