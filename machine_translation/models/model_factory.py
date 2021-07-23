from ..seq2seq.seq2seq_lstm import Seq2SeqLSTM
from ..seq2seq.seq2seq_gru import Seq2SeqGRU
from ..seq2seq.seq2seq_gru_attention import Seq2SeqGRUAtt
# from ..seq2seq.seq2seq_lstm import Seq2SeqAttPadMask
# from ..seq2seq.seq2seq_cnn import Seq2SeqCNN

from machine_translation.utils.init_weights import init_weights_uniform, init_weights_normal


def get_model(
        model_name,
        **kwargs
):
    if model_name == 'seq2seq_lstm':
        model = Seq2SeqLSTM(**kwargs)
    elif model_name == 'seq2seq_gru':
        model = Seq2SeqGRU(**kwargs)
    elif model_name == 'seq2seq_gru_attention':
        model = Seq2SeqGRUAtt(**kwargs)
    # elif model_name == 'seq2seq_gru_attention_pad_mask':
        # model = Seq2SeqAttPadMask(**kwargs)
    # elif model_name == 'seq2seq_cnn':
    #     model = Seq2SeqCNN(**kwargs)
    else:
        raise ValueError(f'{model_name} is not exist!')

    return model


def model_init_weights(
        model_name,
        model
):
    # if model_name == 'seq2seq_lstm':
    #     model = model.apply(init_weights_uniform)
    # elif model_name == 'seq2seq_gru':
    #     model = model.apply(init_weights_normal)
    # elif model_name == 'seq2seq_gru_attention':
    #     model = model.apply(init_weights_normal)
    # elif model_name == 'seq2seq_gru_attention_pad_mask':
    #     model = model.apply(init_weights_normal)
    # elif model_name == 'seq2seq_cnn':
    #     model = model
    # else:
    #     raise ValueError

    return model
