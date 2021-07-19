import random
import torch
import torch.nn as nn

from machine_translation.encoders.encoder_lstm import Encoder
from machine_translation.decoders.decoder_lstm import Decoder


class Seq2SeqLSTM(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            enc_dropout,
            output_dim,
            dec_emb_dim,
            dec_hid_dim,
            dec_dropout,
            n_layers
    ):
        super().__init__()

        self.encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, n_layers, enc_dropout)
        self.decoder = Decoder(output_dim, dec_emb_dim, dec_hid_dim, n_layers, dec_dropout)

        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.n_layers == self.decoder.n_layers, \
            "Encoder and Decoder must have equal number of layers"

    def forward(
            self,
            src,
            trg,
            teacher_forcing_ratio=0.5
    ):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # place predictions in a tensor holding predictions for each token
            outputs[t] = output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            top1 = output.argmax(1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs
