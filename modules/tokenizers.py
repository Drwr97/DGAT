import json
from collections import Counter

class Tokenizer(object):
    def __init__(self, args):
        self.ann_path = args.ann_path
        self.threshold = args.threshold
        self.ann = json.loads(open(self.ann_path, 'r').read())
        self.token2idx, self.idx2token = self.create_vocabulary()

    def create_vocabulary(self):
        total_tokens = []

        for example in self.ann['train']:
            if 'report' in example:
                expression = example['report']
            elif 'expression' in example:
                expression = example['expression']
            else:
                raise KeyError("The expected key ('report' or 'expression') is not found in the dataset.")

            tokens = self.tokenize_expression(expression)
            total_tokens.extend(tokens)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx = {token: idx + 1 for idx, token in enumerate(vocab)}
        idx2token = {idx + 1: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token

    def tokenize_expression(self, expression):
        tokens = expression.split()
        return tokens

    def get_token_by_id(self, id):
        return self.idx2token.get(id, '<unk>')

    def get_id_by_token(self, token):
        return self.token2idx.get(token, self.token2idx['<unk>'])

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, expression):
        tokens = self.tokenize_expression(expression)
        ids = [self.get_id_by_token(token) for token in tokens]
        ids = [self.token2idx.get('<start>', 0)] + ids + [self.token2idx.get('<end>', 0)]
        return ids

    def decode(self, ids):
        tokens = [self.get_token_by_id(idx) for idx in ids if idx > 0]
        expression = ' '.join(tokens) 
        return expression

    def decode_batch(self, ids_batch):
        return [self.decode(ids) for ids in ids_batch]
