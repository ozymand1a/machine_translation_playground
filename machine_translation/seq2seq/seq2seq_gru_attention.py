import random
import torch
import torch.nn as nn

from ..encoders.encoder_gru_attention import Encoder
from ..attention.attention_gru_attention import Attention
from ..decoders.decoder_gru_attention import Decoder


class Seq2SeqGRUAtt(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            enc_dropout,
            output_dim,
            dec_emb_dim,
            dec_hid_dim,
            dec_dropout
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            emb_dim=enc_emb_dim,
            enc_hid_dim=enc_hid_dim,
            dec_hid_dim=dec_hid_dim,
            dropout=enc_dropout
        )

        self.attention = Attention(
            enc_hid_dim=enc_hid_dim,
            dec_hid_dim=dec_hid_dim
        )

        self.decoder = Decoder(
            output_dim=output_dim,
            emb_dim=dec_emb_dim,
            enc_hid_dim=enc_hid_dim,
            dec_hid_dim=dec_hid_dim,
            dropout=dec_dropout,
            attention=self.attention
        )

    def forward(
            self,
            src,
            trg,
            teacher_forcing_ratio=0.5
    ):
        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        # encoder_outputs is all hidden states of the input sequence, back and forwards
        # hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, encoder_outputs)

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
