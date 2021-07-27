from tensorflow.keras import layers
import tensorflow as tf
import sys
import os
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow import keras
from tensorflow.python.keras.constraints import MaxNorm
import tensorflow_addons as tfa
import effnetv2.effnetv2_model as effnetv2_model


tfrecord_trn = '/root/data/hwdb-all/HWDB1.1trn_gnt.tfrecord'
tfrecord_val = '/root/data/hwdb-all/HWDB1.1val_gnt.tfrecord'
tfrecord_tst = '/root/data/hwdb-all/HWDB1.1tst_gnt.tfrecord'
characters_file = '/root/data/hwdb-all/characters.txt'
ckpt_path = '/root/data/hwdb-all/'

BATCH_SIZE = 128
EPOCHS = 30
SHUFFLE_BUFSIZ = 4096
INIT_LEARNING_RATE = 1e-2

IMGSIZ = 224
# IMGSIZ = 64

num_classes = 0


def build_net_EfficientNetB0(n_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=[IMGSIZ, IMGSIZ, 3]),
        effnetv2_model.get_model(
            'efficientnetv2-b0', include_top=False),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(n_classes, activation='softmax'),
    ])
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
    x['image'] = tf.expand_dims(x['image'], axis=-1)
    x['image'] = tf.image.resize(x['image'], (IMGSIZ, IMGSIZ))  # TODO： 试一试
    x['image'] = tf.image.grayscale_to_rgb(x['image'])

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

    model.compile(
        optimizer=tfa.optimizers.RectifiedAdam(learning_rate=INIT_LEARNING_RATE,
                                               min_lr=1e-7,
                                               warmup_proportion=0.15),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])

    # Learning Rate Reducer
    learn_control = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',
                                                         patience=5,
                                                         verbose=1,
                                                         factor=0.2,
                                                         min_lr=1e-7)

    # Checkpoint
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        os.path.join(ckpt_path, 'ckpt-{epoch:d}'),
        monitor='val_accuracy', verbose=1, save_best_only=True,  save_weights_only=True,
        mode='max')

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=ckpt_path, update_freq=100)

    try:
        model.fit(
            trn_ds,
            validation_data=val_ds,
            epochs=EPOCHS, verbose=1,
            callbacks=[learn_control, checkpoint, tb_callback]
        )
    except KeyboardInterrupt:
        print('keras model saved.')
        sys.exit()

    score = model.evaluate(tst_ds, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

    model.save_weights(os.path.join(os.path.dirname(ckpt_path), 'cn_ocr.h5'))


if __name__ == "__main__":
    printDataDir()
    train()

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
