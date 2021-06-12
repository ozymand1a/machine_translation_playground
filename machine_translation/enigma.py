

from machine_translation.utils.train_tools_preparation import Rotor
from machine_translation.models.model_factory import get_model
from machine_translation.utils.seed import set_seed


class Enigma():
    def __init__(self,
                 config_params):
        super(Enigma).__init__()

        self.config_params = config_params
        self.model = get_model(**config_params['base_model'])

    def train(self):
        set_seed(42)

        rotor = Rotor(model_name=self.config_params['model_name'],
                      general_params=self.config_params['general'],
                      checkpoint_callback_params=self.config_params['checkpoint_callback'],
                      logger_params=self.config_params[''],
                      train_params=self.config_params['train_params'])
        trainer = rotor()

        runner = S2SRunner(
            model=self.model,
            train_iterator=train_iterator,
            val_iterator=val_iterator,
            trg_pad_idx=TRG_PAD_IDX,
            clip=params['train_params']['clip'],
            epoch_size=params['train_params']['epoch_size'],
            with_pad=params['with_pad'],
        )

        trainer.fit(runner)

    def test(self):
        pass

    def translate(self, weights):
        pass
