import torch
from pytorch_lightning import seed_everything

from machine_translation.utils.train_tools_preparation import Rotor
from machine_translation.runner import S2SRunner
from machine_translation.dataprepare import TTDataset
from machine_translation.models.model_factory import get_model
from machine_translation.utils.config_processing import get_config
from machine_translation.translator import Translator


class Enigma:
    def __init__(
            self,
            config_params
    ):
        super(Enigma).__init__()
        self.config_params = config_params

        gpu = self.config_params['trainer']['gpus']
        self.device = torch.device(f'cuda:{gpu}' if gpu else 'cpu')

    def load_checkpoint(
            self,
            checkpoint,
            model
    ):
        pretrained_dict = torch.load(checkpoint)

        if 'state_dict' in pretrained_dict.keys():
            pretrained_dict = pretrained_dict['state_dict']

        model_dict = model.state_dict()

        new_state = {}
        for k, v in model_dict.items():
            if 'model.' + k in pretrained_dict.keys() and pretrained_dict['model.' + k].size() == v.size():
                new_state[k] = pretrained_dict['model.' + k]
            else:
                new_state[k] = model_dict[k]
        model.load_state_dict(new_state)
        model = model.to(self.device)

        return model

    def build_loaders(
            self
    ):
        self.tt_dataset = TTDataset(
            data_params=self.config_params['data'],
            device=self.device
        )
        iterator_train, iterator_val, iterator_test = self.tt_dataset()

        return iterator_train, iterator_val, iterator_test

    def build_pad_idxs(
            self
    ):
        trg_pad_idx = self.tt_dataset.get_trg_pad_idx()
        src_pad_idx = self.tt_dataset.get_src_pad_idx()

        return trg_pad_idx, src_pad_idx

    def build_model(
            self
    ):
        input_dim, output_dim = self.tt_dataset.get_dims()

        self.config_params['base_model']['input_dim'] = input_dim
        self.config_params['base_model']['output_dim'] = output_dim

        model = get_model(self.config_params['model_name'], **self.config_params['base_model'])
        model = model.to(self.device)
        print(next(model.parameters()).is_cuda)

        return model

    def build_runner(
            self
    ):
        iterator_train, iterator_val, iterator_test = self.build_loaders()
        trg_pad_idx, src_pad_idx = self.build_pad_idxs()
        model = self.build_model()

        runner = S2SRunner(
            model=model,
            train_iterator=iterator_train,
            val_iterator=iterator_val,
            trg_pad_idx=trg_pad_idx,
            clip=self.config_params['data']['clip'],
            epoch_size=self.config_params['data']['epoch_size'],
            with_pad=self.config_params['data']['with_pad'],
            n_gpu=self.device
        )

        return runner

    def train(
            self
    ):
        seed_everything(42)

        rotor = Rotor(
            model_name=self.config_params['model_name'],
            logging_params=self.config_params['logging'],
            checkpoint_callback_params=self.config_params['checkpoint_callback'],
            trainer_params=self.config_params['trainer']
        )
        trainer = rotor()

        runner = self.build_runner()
        trainer.fit(runner)

    def test(
            self
    ):
        pass

    def translate(
            self,
            checkpoint
    ):
        model = self.build_model()
        model_with_weigths = self.load_checkpoint(
            checkpoint=checkpoint,
            model=model
        )

        translator = Translator(
            model=model_with_weigths,
            device=self.device
        )

        # prepare examples
        data_train,  = self.tt_dataset.data_train
        data_val = self.tt_dataset.data_val
        data_test = self.tt_dataset.data_test

        src = self.tt_dataset.src
        trg = self.tt_dataset.trg

        example_idx = 12

        src_sentence = vars(data_train.examples[example_idx])['src']
        trg_sentence = vars(data_train.examples[example_idx])['trg']

        print(f'src = {src_sentence}')
        print(f'trg = {trg_sentence}')

        # translation = translate_sentence(src, SRC, TRG, model, device)
        translation = translator.translate_sentence(
            src_sentence,
            src,
            trg,
        )

        print(f'predicted trg = {translation}')


if __name__ == '__main__':
    config = 'configs/seq2seq_gru.yaml'
    _, cfg, _ = get_config(config)

    enigma = Enigma(cfg)
    enigma.train()
