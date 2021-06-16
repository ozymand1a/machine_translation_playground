from machine_translation.utils.train_tools_preparation import Rotor
from machine_translation.runner import S2SRunner
from machine_translation.dataset import TTDataset
from machine_translation.models.model_factory import get_model
from machine_translation.utils.seed import set_seed
from machine_translation.utils.config_processing import get_config


class Enigma():
    def __init__(self,
                 config_params):
        super(Enigma).__init__()

        self.config_params = config_params
        self.model = get_model(**config_params['base_model'])

    def build_loaders(self):
        self.tt_dataset = TTDataset(data_params=self.config_params['data'])
        iterator_train, iterator_val, iterator_test = self.tt_dataset()

        return iterator_train, iterator_val, iterator_test

    def build_pad_idxs(self):
        trg_pad_idx = self.tt_dataset.get_trg_pad_idx()
        src_pad_idx = self.tt_dataset.get_src_pad_idx()

        return trg_pad_idx, src_pad_idx

    def build_runner(self):
        iterator_train, iterator_val, iterator_test = self.build_loaders()
        trg_pad_idx, src_pad_idx = self.build_pad_idxs()

        runner = S2SRunner(
            model=self.model,
            train_iterator=iterator_train,
            val_iterator=iterator_val,
            trg_pad_idx=trg_pad_idx,
            clip=self.config_params['data']['clip'],
            epoch_size=self.config_params['data']['epoch_size'],
            with_pad=self.config_params['data']['with_pad'],
        )

        return runner

    def train(self):
        set_seed(42)

        rotor = Rotor(model_name=self.config_params['model_name'],
                      general_params=self.config_params['general'],
                      checkpoint_callback_params=self.config_params['checkpoint_callback'],
                      trainer_params=self.config_params['trainer'])
        trainer = rotor()

        runner = self.build_runner()
        trainer.fit(runner)

    def test(self):
        pass

    def translate(self, weights):
        pass


if __name__ == '__main__':
    config = 'configs/seq2seq_lstm.yaml'
    cfg = get_config(config)

    enigma = Enigma(cfg)
    enigma.train()
