# built in
import os
import pdb
from glob import glob
from functools import partial

# external
import tensorflow as tf
from tqdm import tqdm
#import cv2
import numpy as np

def train_ds(
    path,
    label_path,
    batch_size,
    buffer_size,
    normalize_exams=True,
    output_size=(360, 640, 3),
    augment_options=None,
):
    '''
    generate dataset for training
    Args:
        path: train data path
        batch_size: batch size
        buffer_size: buffer_size
        repeat: should ds be repeated
        slice_types: types of slices to include
        normalize_exams: whether the resulting dataset contain
            the same number of slices from each exam
        output_size: size of images in the dataset
            images will be centrally cropped to match the size
        augment_options (dict): options for augmentation
    '''

    ds = base(
        path,
        label_path,
        output_size=output_size,
    )
    ds = ds.shuffle(buffer_size)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

    print(len(ds))

    return ds


def eval_ds(
    path,
    label_path,
    batch_size,
    output_size=(360, 640, 3),
):
    '''
    generate dataset for training
    Args:
        path: train data path
        batch_size: batch size
        include_meta: whether output ds should contain meta info
        slice_types: types of slices to include
        normalize_exams: whether the resulting dataset contain
            the same number of slices from each exam
        output_size: size of images in the dataset
            images will be centrally cropped to match the size
    '''
    ds = base(
        path,
        label_path,
        output_size=output_size,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    return ds


def predict_ds(path):
    '''
    generate dataset for prediction
    '''
    ds = base(path)
    ds = ds.batch(1)
    return ds

def data_process_after_predict(results):
    rabel=tf.constant([[0,0,0],[61,61,245],[51,221,255],[178,80,80],[36,179,83],[250,50,83],[250,250,55]])
    results = results.map(lambda x: x==tf.expand_dims(tf.math.reduce_max(x,axis=-1),axis=-1))
    results = results.map(lambda x: tf.cast(x, dtype=tf.int32), tf.data.experimental.AUTOTUNE)
    print(results)
    s=iter(results).get_next().shape
    #results = results.map(lambda x: 1 if x==max_results else 0)
    results = results.map(lambda x: tf.math.add_n([tf.expand_dims(x[:,:,i],axis=-1)*tf.reshape(tf.tile(rabel[i],[s[0]*s[1]]),(s[0],s[1],3)) for i in range(len(rabel))]))
    print(len(results))
    print(results)

    return results

def base(path, label_path=None, output_size=(360, 640, 3), dtype=tf.float32):
    '''
    generate base dataset
    '''

    #if not isinstance(path, list): path = list(path)
    #tfrecordsは後日実装
    if os.path.splitext(path[0])[1] == '.tfrecords':
        assert all(map(lambda x: os.path.splitext(x)[1] == '.tfrecords', path))

        #ds = base_from_tfrecords(path, normalize=normalize_exams, include_meta=include_meta, output_slice_types=slice_types)
    else:
        #assert all(map(os.path.isdir, path))
        assert os.path.isdir(path)
        print(path)
        #pattern = list(map(lambda x: os.path.join(x, *'*' * 1), path))
        pattern = os.path.join(path, *'*' * 1)
        #im = tf.data.Dataset.from_tensor_slices(pattern)
        impath = tf.data.Dataset.list_files(pattern)
        #im = tf.data.Dataset.from_tensor_slices(im) いらない？
        im = impath.map(lambda x: tf.io.read_file(x), tf.data.experimental.AUTOTUNE)
        im = im.map(lambda x: tf.io.decode_image(x, channels=3), tf.data.experimental.AUTOTUNE)

        im = im.map(lambda x: tf.reshape(x, [720,1280,3]), tf.data.experimental.AUTOTUNE)
        im = im.map(lambda x: tf.image.resize(x, [360,640]), tf.data.experimental.AUTOTUNE)
        im = im.map(lambda x: tf.cast(x, dtype=dtype), tf.data.experimental.AUTOTUNE)
        im = im.map(lambda x: x / 255.0, tf.data.experimental.AUTOTUNE)

        if label_path is None: return im

        label_pattern = os.path.join(label_path, *'*' * 1)
        label_impath = tf.data.Dataset.list_files(label_pattern)

        assert len(label_impath)==len(impath), "入力画像と教師画像の枚数が異なります"


        #label_im = tf.data.Dataset.range(len(impath))
        #impath.map(lambda x : label_impath.map(lambda y :))[0])
        #base_label_impath = label_impath.map(lambda x : os.path.splitext(os.path.basename(x))[0])
        #base_impath = impath.map(lambda x : os.path.basename(x))

        #base_impath = base_impath.map(lambda x : "black.png" if os.path.splitext(x)[0] not in base_label_impath else x)
        #label_im = base_impath.map(lambda x : os.path.join(label_path, x))
        #label_pattern = list(map(lambda x: os.path.join(x, *'*' * 1), label_path))
        #label_im = tf.data.Dataset.from_tensor_slices(pattern)
        #im = tf.data.Dataset.from_tensor_slices(im)
        label_im = label_impath.map(lambda x: tf.io.read_file(x), tf.data.experimental.AUTOTUNE)
        label_im = label_im.map(lambda x: tf.io.decode_image(x, channels=3), tf.data.experimental.AUTOTUNE)

        label_im = label_im.map(lambda x: tf.reshape(x,[720,1280,3]), tf.data.experimental.AUTOTUNE)
        label_im = label_im.map(lambda x: tf.image.resize(x,[360,640]), tf.data.experimental.AUTOTUNE)

        label_im = label_im.map(lambda x: tf.cast(x, dtype=tf.int32), tf.data.experimental.AUTOTUNE)

        lst = []
        rabel=tf.constant([[0,0,0],[61,61,245],[51,221,255],[178,80,80],[36,179,83],[250,50,83],[250,250,55]])
        cnum=len(rabel)
        print(label_im)
        print(len(label_im))
        #for i in range():
            #for j in range():
        label_im = label_im.map(lambda x: tf.stack([tf.cast(tf.math.abs(x - rabel[i])<5,tf.int32) for i in range(cnum)]))
        print(label_im)
        #iterator = iter(label_im)
        #print(iterator.get_next())
        label_im = label_im.map(lambda x: tf.stack([x[i,:,:,0]*x[i,:,:,1]*x[i,:,:,2] for i in range(cnum)],axis=-1))
        print(label_im)
        print(len(label_im))
        print(label_im)
        iterator = iter(label_im)
        #print(iterator.get_next())
        #print(iterator.get_next())
        #print(tf.shape(label_im))
        assert iterator.get_next().shape==(360,640,7), "shape_error"

        label_im = label_im.map(lambda x: tf.cast(x, dtype=dtype), tf.data.experimental.AUTOTUNE)
        #label_im = label_im.map(lambda x: x / 255.0, tf.data.experimental.AUTOTUNE)

        image_label_ds = tf.data.Dataset.zip((im, label_im))

        print(image_label_ds)


    return image_label_ds



def make_blackimage(
    path,
    label_path,
):
    pattern = os.path.join(path, *'*' * 1)
    label_pattern = os.path.join(label_path, *'*' * 1)
    label_impath = glob(label_pattern)
    im = cv2.imread(label_impath[0])
    im[:,:,:] = 0
    impath = glob(pattern)
    impath = list(map(lambda x : os.path.basename(x),impath))
    fmat = os.path.splitext(os.path.basename(label_impath[0]))[1]
    label_impath = list(map(lambda x : os.path.splitext(os.path.basename(x))[0],label_impath))
    for i in range(len(impath)):
        if os.path.splitext(impath[i])[0] not in label_impath:
            cv2.imwrite(os.path.join(label_path, os.path.splitext(impath[i])[0]) + fmat, im)

    return


def generate_tfrecords(
    path,
    output_path,
    category=None,
    slice_types=('TRA', 'ADC', 'DWI', 'DCEE', 'DCEL', 'label'),
    output_size=(640, 360),
):
    '''
    Generate TFRecords
    Args:
        path: path to the data directory
        output: output path
        category: category to include
            default (None): include all
        slice_types: list of slices to be included
    '''
    def serialize(image, label, ID):
        serialized = tf.py_function(
            lambda image, label, ID:
                tf.train.Example(features=tf.train.Features(feature={
                    'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(image).numpy()])),
                    'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label).numpy()])),
                    'ID': tf.train.Feature(int64_list=tf.train.Int64List(value=[ID])),
                    'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=image.shape)),
                })).SerializeToString(),
            (image, label, ID),
            tf.string,
        )
        return serialized

    pattern = os.path.join(path, *'*' * 1)
    exams = glob(pattern)
    ds = tf.data.Dataset.from_generator(
        lambda: tqdm(map(partial(prepare_combined_slices, slice_types=slice_types), exams), total=len(exams)),
        output_types={
            'slices': tf.uint8,
            'patientID': tf.int64,
            'examID': tf.int64,
            'category': tf.string,
            'path': tf.string,
        },
    )
    ds = ds.map(
        lambda exam_data: {
            'slices': tf.map_fn(
                lambda image: tf.image.crop_to_bounding_box(
                    image,
                    ((tf.shape(image)[:2] - output_size) // 2)[0],
                    ((tf.shape(image)[:2] - output_size) // 2)[1],
                    *output_size,),
                exam_data['slices'],
            ),
            'patientID': exam_data['patientID'],
            'examID': exam_data['examID'],
            'category': exam_data['category'],
            'path': exam_data['path'],
        },
        tf.data.experimental.AUTOTUNE,
    )
    if category is not None: ds = ds.filter(lambda x: x['category'] == category)
    ds = ds.map(
        lambda exam_data: serialize(
            exam_data['slices'],
            exam_data['patientID'],
            exam_data['examID'],
            exam_data['path'],
            exam_data['category'],
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    writer = tf.data.experimental.TFRecordWriter(output, _TFRECORD_COMPRESSION)
    writer.write(ds)
    return
