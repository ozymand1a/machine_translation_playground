import random
import torch
import torch.nn as nn

from ..encoders.encoder_gru_attention_mask_pad import Encoder
from ..attention.attention_gru_attention_mask_pad import Attention
from ..decoders.decoder_gru_attention_mask_pad import Decoder


class Seq2SeqGRUAttMaskPad(nn.Module):
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
            src_pad_idx
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

        self.src_pad_idx = src_pad_idx

    def create_mask(
            self,
            src
    ):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(
            self,
            src,
            src_len,
            trg,
            teacher_forcing_ratio=0.5
    ):
        # src = [src len, batch size]
        # src_len = [batch size]
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
        encoder_outputs, hidden = self.encoder(src, src_len)

        # first input to the decoder is the <sos> tokens
        input = trg[0, :]

        mask = self.create_mask(src)

        # mask = [batch size, src len]

        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            # receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

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
