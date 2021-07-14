from tensorflow.keras import layers
import tensorflow as tf
import sys
import os
from alfred.dl.tf.common import mute_tf
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow import keras
mute_tf()


tfrecord_trn = '/root/data/hwdb_all/HWDB1.1trn_gnt.tfrecord'
tfrecord_val = '/root/data/hwdb_all/HWDB1.1val_gnt.tfrecord'
tfrecord_tst = '/root/data/hwdb_all/HWDB1.1tst_gnt.tfrecord'
characters_file = '/root/data/hwdb_all/characters.txt'
ckpt_path = './root/data/cn_ocr-{epoch}.ckpt'
# BATCH_SIZE = 200
# EPOCHS = 20
BATCH_SIZE = 128
EPOCHS = 15
SHUFFLE_BUFSIZ = 4096
# SHUFFLE_BUFSIZ = 256
# INIT_LEARNING_RATE = 0.0003 # vgg16

# build_net=EfficientNetB0、EfficientNetB6 (unfreeze 20 layers), let use_Transfer_learning = True
INIT_LEARNING_RATE = 2e-4

# build_net=EfficientNetB0_withoutPreTraining、EfficientNetB6_withoutPreTraining, let use_Transfer_learning = False
# INIT_LEARNING_RATE = 1e-3

DECAY_STEP = 1
# DECAY_RATE = 0.9
DECAY_RATE = 0.91
# build_net = build_net_vgg16
# 是否采用迁移学习
# use_Transfer_learning = False
use_Transfer_learning = True

# build_net = build_net_EfficientNetB0_withoutPreTraining
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
    x = layers.Dropout(0.2, name="top_dropout")(x)

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
    a = open(characters_file, 'r').readlines()
    return [i.strip() for i in a]


def lr_scheduler(epoch, lr):
    decay_rate = DECAY_RATE
    decay_step = DECAY_STEP
    if epoch % decay_step == 0 and epoch:
        # return lr * pow(decay_rate, np.floor(epoch / decay_step))
        return lr * decay_rate
    return lr


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
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INIT_LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    callbacks = [tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler, verbose=1)]
    try:
        model.fit(
            trn_ds,
            validation_data=val_ds,
            epochs=EPOCHS, verbose=1,
            callbacks=callbacks
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
    train()
