import spacy

import torch
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator


class TTDataset():
    def __init__(self,
                 data_params,
                 gpu=-1):
        super(TTDataset).__init__()

        self.data_params = data_params
        self.gpu = gpu

    def prepare_fields(self):
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')

        def tokenize_de(text):
            """
            Tokenizes German text from a string into a list of strings (tokens) and reverses it
            """
            return [tok.text for tok in spacy_de.tokenizer(text)][::-1]

        def tokenize_en(text):
            """
            Tokenizes English text from a string into a list of strings (tokens)
            """
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = Field(tokenize=tokenize_de,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        TRG = Field(tokenize=tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)

        return SRC, TRG

    def prepare_datasets(self, SRC, TRG):
        data_train, data_val, data_test = Multi30k.splits(exts=('.de', '.en'),
                                                            fields=(SRC, TRG))

        print(f"Number of training examples: {len(data_train.examples)}")
        print(f"Number of validation examples: {len(data_val.examples)}")
        print(f"Number of testing examples: {len(data_test.examples)}")

        return data_train, data_val, data_test

    def build_vocab(self, SRC, TRG, data_train):
        SRC.build_vocab(data_train, min_freq=2)
        TRG.build_vocab(data_train, min_freq=2)

        return SRC, TRG

    def get_dims(self):
        """
        Should run after __call__
        :return:
        """
        input_dim = len(self.src.vocab)
        output_dim = len(self.trg.vocab)

        return input_dim, output_dim

    def get_src_pad_idx(self):
        src_pad_idx = self.src.vocab.stoi[self.src.pad_token]

        return src_pad_idx

    def get_trg_pad_idx(self):
        trg_pad_idx = self.trg.vocab.stoi[self.trg.pad_token]

        return trg_pad_idx

    def __call__(self):
        src, trg = self.prepare_fields()
        data_train, data_val, data_test = self.prepare_datasets(src, trg)
        self.src, self.trg = self.build_vocab(src, trg, data_train)
        if self.gpu == -1:
            device = torch.device('cpu')
        else:
            device = torch.device(f'cuda:{self.gpu}')

        iterator_train, iterator_val, iterator_test = BucketIterator.splits(
            (data_train, data_val, data_test),
            batch_size=self.data_params['batch_size'],
            device=device)

        return iterator_train, iterator_val, iterator_test
