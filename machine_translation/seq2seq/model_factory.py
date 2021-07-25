from ..seq2seq.seq2seq_lstm import Seq2SeqLSTM
from ..seq2seq.seq2seq_gru import Seq2SeqGRU
from ..seq2seq.seq2seq_gru_attention import Seq2SeqGRUAtt
from ..seq2seq.seq2seq_gru_attention_mask_pad import Seq2SeqGRUAttMaskPad
from ..seq2seq.seq2seq_cnn import Seq2SeqCNN
from ..seq2seq.seq2seq_transformer import Seq2SeqTransformer

from machine_translation.utils.init_weights import init_weights_uniform,\
    init_weights_normal, init_weights_normal_with_constant, initialize_weights_xavier


def get_model(
        model_name,
        src_pad_idx,
        trg_pad_idx,
        device,
        **kwargs
):
    if model_name == 'seq2seq_lstm':
        model = Seq2SeqLSTM(**kwargs)
        model = model.apply(init_weights_uniform)
    elif model_name == 'seq2seq_gru':
        model = Seq2SeqGRU(**kwargs)
        model = model.apply(init_weights_normal)
    elif model_name == 'seq2seq_gru_attention':
        model = Seq2SeqGRUAtt(**kwargs)
        model = model.apply(init_weights_normal_with_constant)
    elif model_name == 'seq2seq_gru_attention_mask_pad':
        model = Seq2SeqGRUAttMaskPad(
            src_pad_idx=src_pad_idx,
            **kwargs
        )
        model = model.apply(init_weights_normal_with_constant)
    elif model_name == 'seq2seq_cnn':
        model = Seq2SeqCNN(
            trg_pad_idx=trg_pad_idx,
            device=device,
            **kwargs
        )
    elif model_name == 'seq2seq_transformer':
        model = Seq2SeqTransformer(
            trg_pad_idx=trg_pad_idx,
            src_pad_idx=src_pad_idx,
            device=device,
            **kwargs
        )
        model = model.apply(initialize_weights_xavier)
    else:
        raise ValueError(f'{model_name} is not exist!')

    return model
