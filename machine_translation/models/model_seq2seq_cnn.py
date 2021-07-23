import random

import torch
import torch.nn as nn
import torch.nn.functional as F



class Seq2SeqCNN(nn.Module):
    def __init__(self,
                 input_dim,
                 enc_emb_dim,
                 enc_hid_dim,
                 enc_layers,
                 enc_kernel_size,
                 enc_dropout,
                 output_dim,
                 dec_emb_dim,
                 dec_hid_dim,
                 dec_layers,
                 dec_kernel_size,
                 dec_dropout,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = Encoder(input_dim, enc_emb_dim, enc_hid_dim, enc_layers, enc_kernel_size, enc_dropout, device)
        self.decoder = Decoder(output_dim, dec_emb_dim, dec_hid_dim, dec_layers, dec_kernel_size, dec_dropout,
                               trg_pad_idx, device)

    def forward(self, src, trg):
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