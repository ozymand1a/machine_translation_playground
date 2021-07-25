import torch
import torch.nn as nn

from machine_translation.encoders.encoder_transformer import Encoder
from machine_translation.decoders.decoder_transformer import Decoder


class Seq2SeqTransformer(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_hid_dim,
            enc_pf_dim,
            enc_dropout,
            output_dim,
            dec_hid_dim,
            dec_pf_dim,
            dec_dropout,
            n_layers,
            n_heads,
            src_pad_idx,
            trg_pad_idx,
            device
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hid_dim=enc_hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=enc_pf_dim,
            dropout=enc_dropout,
            device=device
        )

        self.decoder = Decoder(
            output_dim=output_dim,
            hid_dim=dec_hid_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            pf_dim=dec_pf_dim,
            dropout=dec_dropout,
            device=device
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(
            self,
            src
    ):

        # src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(
            self,
            trg
    ):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(
            self,
            src,
            trg
    ):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention
