# built-in
import pdb
import os
import argparse

# external
import tensorflow as tf
import dsargparse
import yaml

# customs
from .. import engine
from .. import data
from ..utils import dump
from ..utils import load

def predict(
    config,
    save_path,
    data_path,
    ckpt_dir_path,
):
    '''
    Args:
        config (list[str]): configuration file path
            This option accepts arbitrary number of configs.
            If a list is specified, the first one is considered
            as a "main" config, and the other ones will overwrite the content
        save_path: where to save weights/configs/results
        data_path (str): path to the data root dir
        ckpt_dir_path: ckpt directory path to load
    '''
    config = load.load_config(config)
    ds = data.predict_ds(data_path)
    latest = tf.train.latest_checkpoint(ckpt_dir_path)
    model = engine.TFKerasModel(config)
    model.load(latest)
    results = model.predict(ds)
    model.summary()

    results = data.data_process_after_predict(tf.data.Dataset.from_tensor_slices(results))
    results = results.map(lambda x: tf.cast(x, dtype=tf.uint8), tf.data.experimental.AUTOTUNE)

    images = results.map(lambda x: tf.io.encode_jpeg(x, format='rgb'))

    save_path = os.path.join(save_path,"test_images_results")
    iter_images = iter(images)
    for i in range(len(images)):
        tf.io.write_file(os.path.join(save_path,"pre_im_{}.jpg".format(i)),iter_images.get_next())
    #images.map(lambda x: tf.io.write_file(os.path.join(save_path,"pre_im_{}.jpg".format(i+=1),x))
