import spacy

from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator


class TTDataset:
    def __init__(
            self,
            data_params,
            device='cpu'
    ):
        super(TTDataset).__init__()

        self.data_params = data_params
        self.device = device

    def __prepare_fields(self):
        spacy_de = spacy.load('de_core_news_sm')
        spacy_en = spacy.load('en_core_web_sm')

        def tokenize_de(text):
            """
            Tokenizes German text from a string into a list of strings (tokens) and reverses it
            """
            if self.data_params['inversion']:
                return [tok.text for tok in spacy_de.tokenizer(text)][::-1]
            return [tok.text for tok in spacy_de.tokenizer(text)]

        def tokenize_en(text):
            """
            Tokenizes English text from a string into a list of strings (tokens)
            """
            return [tok.text for tok in spacy_en.tokenizer(text)]

        SRC = Field(
            tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True
        )

        TRG = Field(
            tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True
        )

        return SRC, TRG

    def __prepare_datasets(
            self,
            SRC,
            TRG
    ):
        self.data_train, self.data_val, self.data_test = Multi30k.splits(
            exts=('.de', '.en'),
            fields=(SRC, TRG)
        )

        print(f"Number of training examples: {len(self.data_train.examples)}")
        print(f"Number of validation examples: {len(self.data_val.examples)}")
        print(f"Number of testing examples: {len(self.data_test.examples)}")

    def build_vocab(
            self,
            SRC,
            TRG,
    ):
        SRC.build_vocab(self.data_train, min_freq=2)
        TRG.build_vocab(self.data_train, min_freq=2)

        print(f"Unique tokens in source (de) vocabulary: {len(SRC.vocab)}")
        print(f"Unique tokens in target (en) vocabulary: {len(TRG.vocab)}")

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
        src, trg = self.__prepare_fields()
        self.__prepare_datasets(src, trg)
        self.src, self.trg = self.build_vocab(
            src,
            trg
        )

        iterator_train, iterator_val, iterator_test = BucketIterator.splits(
            (self.data_train, self.data_val, self.data_test),
            batch_size=self.data_params['batch_size'],
            device=self.device
        )

        return iterator_train, iterator_val, iterator_test
