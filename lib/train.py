from keras import optimizers
from keras.callbacks import EarlyStopping, LearningRateScheduler, Callback


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


def define_callbacks(early_stopping_delta, early_stopping_epochs, use_lr_strategy=True, initial_lr=0.005, lr_drop_koef=0.66, epochs_to_drop=5, model_checkpoint_dir=None):
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=early_stopping_delta, patience=early_stopping_epochs, verbose=1)
    callbacks_list = [early_stopping]
    if model_checkpoint_dir is not None:
        model_checkpoint = ModelCheckpoint(os.path.join(model_checkpoint_dir,'weights.{epoch:02d}-{val_loss:.2f}.h5'), monitor='val_loss', save_best_only=True, verbose=0)
        callbacks_list.append(model_checkpoint)
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
