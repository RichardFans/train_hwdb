from tensorflow.keras import layers
import tensorflow as tf
import sys
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow import keras
from tensorflow.python.keras.callbacks import LearningRateScheduler
from tensorflow.python.keras.constraints import MaxNorm


tfrecord_trn = '/root/data/hwdb-all/HWDB1.1trn_gnt.tfrecord'
tfrecord_val = '/root/data/hwdb-all/HWDB1.1val_gnt.tfrecord'
tfrecord_tst = '/root/data/hwdb-all/HWDB1.1tst_gnt.tfrecord'
characters_file = '/root/data/hwdb-all/characters.txt'
ckpt_path = '/root/data/hwdb-all/cn_ocr-{epoch}.ckpt'

BATCH_SIZE = 256
EPOCHS = 15
SHUFFLE_BUFSIZ = 4096
INIT_LEARNING_RATE = 1e-2

# 大概1/5 EPOCHS
DECAY_STEP = 1024
DECAY_RATE = 0.97
# 是否采用迁移学习
# use_Transfer_learning = False
use_Transfer_learning = True

# IMGSIZ = 224
IMGSIZ = 64

num_classes = 0


def build_net_EfficientNetB0(n_classes):
    base_model = EfficientNetB0(weights='imagenet',
                                input_shape=(IMGSIZ, IMGSIZ, 3),
                                include_top=False)
    base_model.trainable = False

    inputs = keras.Input(shape=(IMGSIZ, IMGSIZ, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    # x = layers.Dense(4096, kernel_constraint=MaxNorm(3.))(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.5, name="top_dropout")(x)

    outputs = layers.Dense(n_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model


build_net = build_net_EfficientNetB0


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
    if use_Transfer_learning:
        x['image'] = tf.expand_dims(x['image'], axis=-1)
        x['image'] = tf.image.grayscale_to_rgb(x['image'])
    else:
        x['image'] = tf.expand_dims(x['image'], axis=-1)
    x['image'] = tf.image.resize(x['image'], (IMGSIZ, IMGSIZ))  # TODO： 试一试
    x['image'] = x['image'] / 255.
    # x['label'] = tf.one_hot(x['label'], num_classes)
    return x['image'], x['label']


def load_ds(ds_path):
    ds = tf.data.TFRecordDataset([ds_path], compression_type="ZLIB")
    ds = ds.map(parse_example)
    ds = ds.shuffle(SHUFFLE_BUFSIZ).map(preprocess).batch(BATCH_SIZE)
    return ds


def load_characters():
    a = open(characters_file, 'r', encoding='utf8').readlines()
    return [i.strip() for i in a]


def printDataDir():
    f = []
    print(os.path.dirname(ckpt_path))
    print('Data dir Path: ', os.path.dirname(ckpt_path))
    for (dirpath, dirnames, filenames) in os.walk(os.path.dirname(ckpt_path)):
        f.extend(filenames)
        break
    print('File in data dir: ', f)


def train():
    all_characters = load_characters()
    num_classes = len(all_characters)
    print('all characters: {}'.format(num_classes))
    trn_ds = load_ds(tfrecord_trn)
    val_ds = load_ds(tfrecord_val)
    tst_ds = load_ds(tfrecord_tst)

    model = build_net(num_classes)
    model.summary()
    print('model loaded.')

    # The answer, in a nutshell
    # If your targets are one-hot encoded, use categorical_crossentropy.
    # But if your targets are integers, use sparse_categorical_crossentropy

    if use_Transfer_learning:
        for layer in model.layers[-20:]:
            print('===>', layer)
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        INIT_LEARNING_RATE, decay_steps=DECAY_STEP, decay_rate=DECAY_RATE, staircase=True
    )

    class CustomCallback(tf.keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()

        def on_train_batch_begin(self, batch, logs=None):
            lr = tf.keras.backend.get_value(
                self.model.optimizer.lr(self.model.optimizer.iterations)
            )
            print('======> lr:\n', lr)

    # def lr_schedule2(epoch):
    #     # lr = model.optimizer.lr(model.optimizer.iterations).numpy()
    #     # # lr = backend.get_value(model.optimizer.lr)

    #     lr = tf.keras.backend.get_value(
    #         model.optimizer.lr(model.optimizer.iterations)
    #     )
    #     print('======> lr:\n', lr, epoch)

    #     return lr

    # lrcb = LearningRateScheduler(lr_schedule2)
    lrcb = CustomCallback()

    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=lr_schedule, momentum=0.9, nesterov=True),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    try:
        model.fit(
            trn_ds,
            validation_data=val_ds,
            epochs=5, verbose=1,
            callbacks=[lrcb]
        )
    except KeyboardInterrupt:
        # model.save_weights(ckpt_path.format(epoch=0))
        print('keras model saved.')
        sys.exit()

    score = model.evaluate(tst_ds, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights(ckpt_path.format(epoch=0))
    model.save(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))


if __name__ == "__main__":
    printDataDir()
    train()
