import torch.nn as nn

from machine_translation.encoders.encoder_cnn import Encoder
from machine_translation.decoders.decoder_cnn import Decoder


class Seq2SeqCNN(nn.Module):
    def __init__(
            self,
            input_dim,
            enc_emb_dim,
            enc_hid_dim,
            enc_dropout,
            enc_kernel_size,
            output_dim,
            dec_emb_dim,
            dec_hid_dim,
            dec_dropout,
            dec_kernel_size,
            n_layers,
            trg_pad_idx,
            device
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            emb_dim=enc_emb_dim,
            hid_dim=enc_hid_dim,
            n_layers=n_layers,
            kernel_size=enc_kernel_size,
            dropout=enc_dropout,
            device=device
        )

        self.decoder = Decoder(
            output_dim=output_dim,
            emb_dim=dec_emb_dim,
            hid_dim=dec_hid_dim,
            n_layers=n_layers,
            kernel_size=dec_kernel_size,
            dropout=dec_dropout,
            trg_pad_idx=trg_pad_idx
        )

    def forward(
            self,
            src,
            trg
    ):
        # src = [batch size, src len]
        # trg = [batch size, trg len - 1] (<eos> token sliced off the end)

        # calculate z^u (encoder_conved) and (z^u + e) (encoder_combined)
        # encoder_conved is output from final encoder conv. block
        # encoder_combined is encoder_conved plus (elementwise) src embedding plus
        #  positional embeddings
        encoder_conved, encoder_combined = self.encoder(src)

        # encoder_conved = [batch size, src len, emb dim]
        # encoder_combined = [batch size, src len, emb dim]

        # calculate predictions of next words
        # output is a batch of predictions for each word in the trg sentence
        # attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg, encoder_conved, encoder_combined)

        # output = [batch size, trg len - 1, output dim]
        # attention = [batch size, trg len - 1, src len]

        return output, attention
