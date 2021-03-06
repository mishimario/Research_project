'''
provides various functions to calculate loss
'''

# built-in
import os
import sys
import pdb

# external
import tensorflow as tf
import tensorflow_addons as tfa

# custom


@tf.function
def tf_weighted_crossentropy(label, pred, weight=None, weight_add=0, weight_mul=1, from_logits=False):
    '''
    calculates weighted loss
    '''
    if tf.shape(label)[0] == 0:
        return tf.zeros([0], dtype=pred.dtype)

    if weight is None:
        positive_rate = tf_get_positive_rate(label)
        #weight = 1 / positive_rate if positive_rate > 0.0 else 1.0
        positive_rate =  positive_rate + tf.cast(positive_rate==0,positive_rate.dtype)
        weight = 1 / positive_rate

    weight = weight_mul * weight + weight_add

    weight = tf.expand_dims(weight,-1)
    weight = tf.expand_dims(weight,-1)

    with tf.control_dependencies([tf.debugging.assert_greater_equal(weight, 0.0, name='assert_on_weight')]):
        weight_mask = label * weight + tf.cast(label==0,label.dtype)
        #print(weight_mask.shape)

        #+ tf.cast(tf.stack([tf.map_fn(fn=lambda x: 1 if x==0 else 0,elems=elem) for elem in tf.unstack(label)]),label.dtype)
    #print(weight.shape)
    label = tf.expand_dims(label, -1)
    pred = tf.expand_dims(pred, -1)
    bce = tf.keras.losses.BinaryCrossentropy(reduction=tf.losses.Reduction.NONE, from_logits=from_logits)
    loss = bce(label, pred, sample_weight=weight_mask)
    #print(loss.shape)
    loss = tf.reduce_mean(loss, [1, 2])
    #print(loss.shape)
    return loss


class TFWeightedCrossentropy(tf.keras.losses.Loss):
    def __init__(
            self,
            weight=None,
            weight_add=0.0,
            weight_mul=1.0,
            label_smoothing=False,
            label_smoothing_filter_size=6,
            label_smoothing_sigma=3,
    ):
        self.weight = weight
        self.weight_add = weight_add
        self.weight_mul = weight_mul
        self.label_smoothing = label_smoothing
        self.label_smoothing_filter_size = label_smoothing_filter_size
        self.label_smoothing_sigma = label_smoothing_sigma

        super().__init__(name='weighted_crossentropy')
        return

    def call(self, y_true, y_pred):
        y_pred_logits = y_pred._keras_logits
        _, _, _, a=y_pred_logits.shape
        if self.label_smoothing:
            y_true = tf.expand_dims(y_true, -1)
            y_true = tfa.image.gaussian_filter2d(
                y_true, filter_shape=self.label_smoothing_filter_size, sigma=self.label_smoothing_sigma,
            )
            y_true = tf.squeeze(y_true, -1)
        loss = tf.stack([tf_weighted_crossentropy(
            y_true[:,:,:,i], y_pred_logits[:,:,:,i], from_logits=True,
            weight=self.weight, weight_add=self.weight_add, weight_mul=self.weight_mul,
        ) for i in range(a)])
        return tf.reduce_mean(loss,0)

    def get_config(self):
        config = super().get_config()
        config.update(dict(
            weight=self.weight,
            weight_add=self.weight_add,
            weight_mul=self.weight_mul,
            label_smoothing=self.label_smoothing,
            label_smoothing_filter_size=self.label_smoothing_filter_size,
            label_smoothing_sigma=self.label_smoothing_sigma,
        ))
        return config


def tf_get_positive_rate(label):
    max_value = tf.reduce_max(label)
    min_value = tf.reduce_min(label)

    upbound = tf.debugging.assert_less_equal(max_value, 1.0, name='assert_on_max')
    lowbound = tf.debugging.assert_greater_equal(min_value, 0.0, name='assert_on_min')

    with tf.control_dependencies([upbound, lowbound]):
        positive_rate = tf.reduce_sum(label,[1,2]) / tf.cast(tf.reduce_prod(tf.shape(label)[1:]), tf.float32)

    assert_range = [
        tf.debugging.assert_greater_equal(positive_rate, 0.0, name='assert_on_range_lower'),
        tf.debugging.assert_less_equal(positive_rate, 1.0, name='assert_on_range_higher'),
    ]
    with tf.control_dependencies(assert_range):
        return positive_rate


tf.keras.utils.get_custom_objects().update(weighted_crossentropy=tf_weighted_crossentropy)
tf.keras.utils.get_custom_objects().update(WeightedCrossentropy=TFWeightedCrossentropy)
