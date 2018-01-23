import json
from math import pow, floor

from keras import optimizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback

try:
    import nirvana_dl
except ImportError:
    pass


def step_decay(initial_lr, lr_drop_koef, epochs_to_drop, epoch):
    return initial_lr * pow(lr_drop_koef, floor((1 + epoch) / epochs_to_drop))


class LossHistory(Callback):
    def __init__(self, initial_lr, lr_drop_koef, epochs_to_drop):
        self.initial_lr = initial_lr
        self.lr_drop_koef = lr_drop_koef
        self.epochs_to_drop = epochs_to_drop

    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(self.initial_lr, self.lr_drop_koef, self.epochs_to_drop, len(self.losses)))


def define_callbacks(early_stopping_delta, early_stopping_epochs, use_lr_strategy=True, initial_lr=0.005, lr_drop_koef=0.66, epochs_to_drop=5):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta, patience=early_stopping_epochs, verbose=1)
    callbacks_list = [early_stopping]
    if use_lr_strategy:
        epochs_to_drop = float(epochs_to_drop)
        loss_history = LossHistory(initial_lr, lr_drop_koef, epochs_to_drop)
        lrate = LearningRateScheduler(lambda epoch: step_decay(initial_lr, lr_drop_koef, epochs_to_drop, epoch))
        callbacks_list.append(loss_history)
        callbacks_list.append(lrate)
    return callbacks_list


def train(x_train, y_train, model, batch_size, num_epochs, learning_rate=0.001, early_stopping_delta=0.0, early_stopping_epochs=10, use_lr_strategy=True, lr_drop_koef=0.66, epochs_to_drop=5, logger=None):
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
    if logger is not None:
        model.summary(print_fn=lambda line: logger.debug(line))
    else:
        model.summary()

    callbacks_list = define_callbacks(early_stopping_delta,
                                    early_stopping_epochs,
                                    use_lr_strategy=use_lr_strategy,
                                    initial_lr=learning_rate,
                                    lr_drop_koef=lr_drop_koef,
                                    epochs_to_drop=epochs_to_drop)

    hist = model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=callbacks_list,
                     validation_split=0.1,
                     shuffle=True,
                     verbose=1)
    return hist


def continue_train(x_train, y_train, model, batch_size, num_epochs, learning_rate_decay, learning_rate=0.001, early_stopping_delta=0.0, early_stopping_iters=10, use_lr_strategy=True, lr_drop_koef=0.66, epochs_to_drop=5):
    callbacks_list = define_callbacks(early_stopping_delta,
                                    early_stopping_iters,
                                    use_lr_strategy=use_lr_strategy,
                                    initial_lr=learning_rate,
                                    lr_drop_koef=lr_drop_koef,
                                    epochs_to_drop=epochs_to_drop)

    hist = model.fit(x_train,
                     y_train,
                     batch_size=batch_size,
                     epochs=num_epochs,
                     callbacks=callbacks_list,
                     validation_split=0.1,
                     shuffle=True,
                     verbose=1)
    return hist


class Params(object):
    def __init__(self, config=None):
        self._params = self._common_init()
        config_params = self._load_from_file(config)
        self._update_params(config_params)

    def _load_from_file(self, fname):
        if fname is None:
            return {}
        elif fname == 'nirvana':
            return nirvana_dl.params()
        with open(fname) as f:
            return json.loads(f.read())

    def _common_init(self):
        common_params = {
                    'warm_start': False,
                    'model_file': None,
                    'batch_size': 256,
                    'num_epochs': 10,
                    'learning_rate': 0.0001,
                    'early_stopping_delta': 0.001,
                    'early_stopping_epochs': 2,
                    'use_lr_strategy': True,
                    'lr_drop_koef': 0.5,
                    'epochs_to_drop': 1,
                    'l2_weight_decay':0.0001,
                    'dropout_val': 0.5,
                    'dense_dim': 32,
                    'train_embeds': False}

        params = {'models': [],
                  'dense': common_params,
                  'cnn': common_params,
                  'lstm': common_params,
                  'concat': common_params}

        params['dense']['dense_dim'] = 100
        params['dense']['n_layers'] = 10
        params['dense']['concat'] = 0
        params['dense']['pool'] = 'max'
        params['cnn']['num_filters'] = 64
        params['lstm']['lstm_dim'] = 50
        params['concat']['num_filters'] = 64
        params['concat']['lstm_dim'] = 50
        params['concat']['n_layers'] = 10
        params['concat']['concat'] = 0
        params['concat']['pool'] = 'max'

        params['catboost'] = {
                    'add_bow': False,
                    'bow_top': 100,
                    'iterations': 1000,
                    'depth': 6,
                    'rsm': 1,
                    'learning_rate': 0.01,
                    'device_config': None}
        return params

    def _update_params(self, params):
        if params is not None and params:
            for key in params.keys():
                if isinstance(params[key], dict):
                    self._params.setdefault(key, {})
                    self._params[key].update(params[key])
                else:
                    self._params[key] = params[key]

    def get(self, key):
        return self._params.get(key, None)
