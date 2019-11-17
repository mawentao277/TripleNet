from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from evaluate import evaluate_ubuntu, evaluate_douban
import codecs
import json
import os

class SaveModelCallback(Callback):
    def __init__(self, args, single_model):
        """
        :param single_model: keras can only save single gpu model, not parallel model
        """
        super().__init__()
        self.epoch_counter = 0
        self.batch_counter = 0
        self.seed = 0
        self.single_model = single_model
        self.config = args

    def on_epoch_begin(self, epoch, logs={}):
        if self.epoch_counter == 0:
            # we save config file at first epoch
            with codecs.open(os.path.join(self.config.output_dir, 'config.json'), 'w',
                             encoding='utf-8') as f:
                json.dump(self.config.__dict__, f)

            if self.config.task == 'ubuntu':
                result = evaluate_ubuntu(self.config, self.single_model)
            else:
                result = evaluate_douban(self.config, self.single_model)
            self.single_model.save_weights(
                self.config.output_dir + 'model_epoch' + str(self.epoch_counter) + '_prec' +
                str(result) + '.hdf5', overwrite=True
            )
        self.epoch_counter += 1

    def on_batch_begin(self, batch, logs={}):
        self.batch_counter += 1
        if self.config.task == 'ubuntu' and self.batch_counter % 3125 == 0 and self.epoch_counter >= 3:
            # we will eval per 3125 steps
            result = evaluate_ubuntu(self.config, self.single_model)
            self.single_model.save_weights(
                self.config.output_dir + 'model_epoch' + str(self.epoch_counter) + '_prec' +
                str(result) + '.hdf5', overwrite=True)

        if self.config.task == 'douban' and self.batch_counter % 2000 == 0:
            result = evaluate_douban(self.config, self.single_model)
            self.single_model.save_weights(
                self.config.output_dir + 'model_epoch' + str(self.epoch_counter) + '_prec' +
                str(result) + '.hdf5', overwrite=True)
