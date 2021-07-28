import copy
import re
from tensorflow.keras import backend, layers
import tensorflow as tf
import sys
import os
from tensorflow import keras
import tensorflow_addons as tfa
from absl import app, logging
import effnetv2.effnetv2_model as effnetv2_model
import effnetv2.datasets as datasets
import effnetv2.hparams as hparams
import effnetv2.effnetv2_configs as effnetv2_configs
import effnetv2.utils as utils

useFeaturize = os.path.isdir('/home/featurize/')
if useFeaturize:
    datapath = '/home/featurize/data/'
    savepath = '/home/featurize/work/'
else:
    savepath = datapath = '/root/data/hwdb-all/'


tfrecord_trn = "{}HWDB1.1trn_gnt.tfrecord".format(datapath)
tfrecord_val = "{}HWDB1.1val_gnt.tfrecord".format(datapath)
tfrecord_tst = "{}HWDB1.1tst_gnt.tfrecord".format(datapath)
characters_file = "{}characters.txt".format(datapath)
ckpt_path = savepath

# Currently, supported model_name includes:
# efficientnetv2-s, efficientnetv2-m, efficientnetv2-l, efficientnetv2-b0,
# efficientnetv2-b1, efficientnetv2-b2, efficientnetv2-b3.
# We also support all EfficientNetV1 models including:
# efficientnet-b0/b1/b2/b3/b4/b5/b6/b7/b8/l2
config = hparams.Config(
    model=dict(
        model_name='efficientnetv2-s'
    ),
    train=dict(
        batch_size=1024,
        isize=224
    ),
    eval=dict(
        isize=224,  # image size
    ),
    runtime=dict(
        strategy='gpu'
    ),
    data=dict(
        ds_name='hwdbcasia'
    )
)


def parse_example(record):
    features = tf.io.parse_single_example(record,
                                          features={
                                              'label':
                                                  tf.io.FixedLenFeature(
                                                      [], tf.int64),
                                              'image':
                                                  tf.io.FixedLenFeature(
                                                      [], tf.string),
                                          })
    img = tf.io.decode_raw(features['image'], out_type=tf.uint8)
    img = tf.cast(tf.reshape(img, (64, 64)), dtype=tf.float32)
    label = tf.cast(features['label'], tf.int64)
    return {'image': img, 'label': label}


def preprocess(x):
    x['image'] = tf.expand_dims(x['image'], axis=-1)
    if config.train.isize != 64:
        x['image'] = tf.image.resize(
            x['image'], (config.train.isize, config.train.isize))
    x['image'] = tf.image.grayscale_to_rgb(x['image'])

    x['image'] = x['image'] / 255.
    x['label'] = tf.one_hot(x['label'], config.model.num_classes)
    return x['image'], x['label']


def load_ds(ds_path, repeat=True):
    ds = tf.data.TFRecordDataset([ds_path], compression_type="ZLIB")
    ds = ds.map(parse_example)
    ds = ds.shuffle(16 * 1024).map(preprocess).batch(config.train.batch_size)
    if repeat:
        ds = ds.repeat()
    return ds


def load_characters():
    a = open(characters_file, 'r', encoding='utf8').readlines()
    return [i.strip() for i in a]


def printDataDir():
    f = []
    logging.info(os.path.dirname(ckpt_path))
    logging.info('Data dir Path: %s', os.path.dirname(ckpt_path))
    for (dirpath, dirnames, filenames) in os.walk(os.path.dirname(ckpt_path)):
        f.extend(filenames)
        break
    logging.info('File in data dir: %s', f)


def build_tf2_optimizer(learning_rate,
                        optimizer_name='rmsprop',
                        decay=0.9,
                        epsilon=0.001,
                        momentum=0.9):
    """Build optimizer."""
    if optimizer_name == 'sgd':
        logging.info('Using SGD optimizer')
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == 'momentum':
        logging.info('Using Momentum optimizer')
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        logging.info('Using RMSProp optimizer')
        optimizer = tf.keras.optimizers.RMSprop(learning_rate, decay, momentum,
                                                epsilon)
    elif optimizer_name == 'adam':
        logging.info('Using Adam optimizer')
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    else:
        raise Exception('Unknown optimizer: %s' % optimizer_name)

    return optimizer


class TrainableModel(effnetv2_model.EffNetV2Model):
    """Wraps efficientnet to make a keras trainable model.

    Handles efficientnet's multiple outputs and adds weight decay.
    """

    def __init__(self,
                 model_name='efficientnetv2-s',
                 model_config=None,
                 name=None,
                 weight_decay=0.0):
        super().__init__(
            model_name=model_name,
            model_config=model_config,
            name=name or model_name)

        self.weight_decay = weight_decay

    def _reg_l2_loss(self, weight_decay, regex=r'.*(kernel|weight):0$'):
        """Return regularization l2 loss loss."""
        var_match = re.compile(regex)
        return weight_decay * tf.add_n([
            tf.nn.l2_loss(v)
            for v in self.trainable_variables
            if var_match.match(v.name)
        ])

    def train_step(self, data):
        images, labels = data

        with tf.GradientTape() as tape:
            pred = self(images, training=True)
            pred = tf.cast(pred, tf.float32)
            loss = self.compiled_loss(
                labels,
                pred,
                regularization_losses=[self._reg_l2_loss(self.weight_decay)])

        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(labels, pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        images, labels = data

        pred = self(images, training=False)
        pred = tf.cast(pred, tf.float32)

        self.compiled_loss(
            labels,
            pred,
            regularization_losses=[self._reg_l2_loss(self.weight_decay)])

        self.compiled_metrics.update_state(labels, pred)
        return {m.name: m.result() for m in self.metrics}


def main(_) -> None:
    cfg = copy.deepcopy(hparams.base_config)
    cfg.override(effnetv2_configs.get_model_config(config.model.model_name))
    cfg.override(datasets.get_dataset_config(config.data.ds_name))
    cfg.model.num_classes = cfg.data.num_classes
    cfg.override(config)
    config.update(cfg)

    # load data
    # all_characters = load_characters()
    trn_ds = load_ds(tfrecord_trn)
    val_ds = load_ds(tfrecord_val)
    tst_ds = load_ds(tfrecord_tst, repeat=False)

    strategy = config.runtime.strategy
    if strategy == 'tpu' and not config.model.bn_type:
        config.model.bn_type = 'tpu_bn'

    # log and save config.
    logging.info('config=%s', str(config))
    config.save_to_yaml(os.path.join(ckpt_path, 'config.yaml'))

    # 暂时不管TPU
    if strategy == 'gpus':
        ds_strategy = tf.distribute.MirroredStrategy()
        logging.info('All devices: %s', tf.config.list_physical_devices('GPU'))
    else:
        if tf.config.list_physical_devices('GPU'):
            ds_strategy = tf.distribute.MirroredStrategy(['GPU:0'])
        else:
            ds_strategy = tf.distribute.MirroredStrategy(['CPU:0'])

    with ds_strategy.scope():
        train_split = config.train.split or 'train'
        eval_split = config.eval.split or 'eval'
        num_train_images = config.data.splits[train_split].num_images
        num_eval_images = config.data.splits[eval_split].num_images

        train_size = config.train.isize
        eval_size = config.eval.isize

        if config.runtime.mixed_precision:
            image_dtype = 'bfloat16' if strategy == 'tpu' else 'float16'
            precision = 'mixed_bfloat16' if strategy == 'tpu' else 'mixed_float16'
            policy = tf.keras.mixed_precision.Policy(precision)
            tf.keras.mixed_precision.set_global_policy(policy)

        model = TrainableModel(
            config.model.model_name,
            config.model,
            weight_decay=config.train.weight_decay)

        if config.train.ft_init_ckpt:  # load pretrained ckpt for finetuning.
            model(tf.keras.Input([None, None, 3]))
            ckpt = config.train.ft_init_ckpt
            utils.restore_tf2_ckpt(
                model, ckpt, exclude_layers=('_fc', 'optimizer'))

        steps_per_epoch = num_train_images // config.train.batch_size
        total_steps = steps_per_epoch * config.train.epochs

        scaled_lr = config.train.lr_base * (config.train.batch_size / 256.0)
        scaled_lr_min = config.train.lr_min * (config.train.batch_size / 256.0)

        logging.info("Initial Learning Rate: %f", scaled_lr)

        learning_rate = utils.WarmupLearningRateSchedule(
            scaled_lr,
            steps_per_epoch=steps_per_epoch,
            decay_epochs=config.train.lr_decay_epoch,
            warmup_epochs=config.train.lr_warmup_epoch,
            decay_factor=config.train.lr_decay_factor,
            lr_decay_type=config.train.lr_sched,
            total_steps=total_steps,
            minimal_lr=scaled_lr_min)

        optimizer = build_tf2_optimizer(
            learning_rate, optimizer_name=config.train.optimizer)

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=config.train.label_smoothing, from_logits=True),
            metrics=[
                tf.keras.metrics.TopKCategoricalAccuracy(k=1, name='acc_top1'),
                tf.keras.metrics.TopKCategoricalAccuracy(k=5, name='acc_top5')
            ],
        )

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            os.path.join(ckpt_path, 'ckpt-{epoch:d}'), verbose=1,  save_weights_only=True)

        tb_callback = tf.keras.callbacks.TensorBoard(
            log_dir=ckpt_path, update_freq=100)

        class LearningRateLogCallback(tf.keras.callbacks.Callback):
            def __init__(self) -> None:
                super().__init__()
                self.freq = steps_per_epoch // 10
                self.step = 0

            def on_epoch_begin(self, epoch, logs):
                self.epoch = epoch

            def on_train_batch_begin(self, batch, logs=None):
                if batch % self.freq == 0:
                    lr = tf.keras.backend.get_value(
                        self.model.optimizer.lr(
                            self.model.optimizer.iterations)
                    )
                    tf.summary.scalar('Learning rate', data=lr, step=self.step)
                    self.step += 1

        lr_log_callback = LearningRateLogCallback()

        try:
            model.fit(
                trn_ds,
                epochs=config.train.epochs,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_ds,
                validation_steps=num_eval_images // config.eval.batch_size,
                verbose=1,
                callbacks=[ckpt_callback, tb_callback, lr_log_callback]
            )
        except KeyboardInterrupt:
            sys.exit()

        score = model.evaluate(tst_ds, verbose=0)
        logging.info('Test score: %f', score[0])
        logging.info('Test accuracy: %f', score[1])

        model.save_weights(os.path.join(
            os.path.dirname(ckpt_path), 'cn_ocr.h5'))


if __name__ == "__main__":
    app.run(main)

# if __name__ == "__main__":
#     all_characters = load_characters()
#     num_classes = len(all_characters)
#     print('all characters: {}'.format(num_classes))
#     tst_ds = load_ds(tfrecord_tst)

#     model = build_net(num_classes)

#     model.compile(
#         optimizer=tfa.optimizers.RectifiedAdam(learning_rate=INIT_LEARNING_RATE,
#                                                min_lr=1e-7,
#                                                warmup_proportion=0.15),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=['accuracy'])

#     ckpt = tf.train.latest_checkpoint(ckpt_path)
#     _ = model(tf.ones([1, 64, 64, 3]), training=False)
#     model.load_weights(ckpt)
#     model.load_weights(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))

#     score = model.evaluate(tst_ds, verbose=0)
#     print('Test score:', score[0])
#     print('Test accuracy:', score[1])

    # model.save_weights(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))
