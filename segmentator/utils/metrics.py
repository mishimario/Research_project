'''
provide custom metrics
'''

# built-in
import pdb
import os
from multiprocessing import cpu_count

# external
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

# custom
#from . import image as custom_image_ops


def solve_metric(metric_spec):
    '''
    solve metric spec and return metric instance
    '''
    if isinstance(metric_spec, str): return metric_spec
    elif isinstance(metric_spec, dict):
        assert len(metric_spec) == 1
        pass
    else: raise ValueError

    metric_name, metric_options = list(metric_spec.items())[0]
    instance = tf.keras.metrics.get({
        "class_name": metric_name,
        "config": metric_options,
    })
    return instance


class FBetaScore(tf.keras.metrics.Metric):
    def __init__(self, beta, thresholds, epsilon=1e-07, **kargs):
        super().__init__(**kargs)
        assert beta > 0
        self.beta = beta
        self.epsilon = epsilon
        self.thresholds = thresholds
        self.precision = None
        self.recall = None
        self.n_label = 0
        #self.prepare_precision_recall()
        return

    def prepare_precision_recall(self):
        self.precision = [tf.keras.metrics.Precision(thresholds=self.thresholds) for _ in range(self.n_label)]
        self.recall = [tf.keras.metrics.Recall(thresholds=self.thresholds) for _ in range(self.n_label)]
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        #y_true = tf.transpose(y_true, perm=[1,2,3,0])
        #y_pred = tf.transpose(y_pred, perm=[1,2,3,0])
        if self.precision is None or self.recall is None:
            _, _, _, self.n_label = y_true.shape
            self.prepare_precision_recall()
            #self.precision = tf.map_fn(fn=lambda _:tf.keras.metrics.Precision(thresholds=self.thresholds), elems=y_true)
            #self.recall = tf.map_fn(fn=lambda _:tf.keras.metrics.Recall(thresholds=self.thresholds), elems=y_true)

        for i in range(self.n_label):
            self.precision[i].update_state(y_true[:,:,:,i], y_pred[:,:,:,i], sample_weight=sample_weight)
            self.recall[i].update_state(y_true[:,:,:,i], y_pred[:,:,:,i], sample_weight=sample_weight)
        #tf.map_fn(fn=lambda x: x[0].update_state(x[1], x[2], sample_weight=sample_weight),elems=[self.precision, y_true, y_pred])
        #self.precision.update_state(y_true, y_pred, sample_weight=sample_weight)
        #self.recall.update_state(y_true, y_pred, sample_weight=sample_weight)
        return

    def result(self):
        precision =tf.stack([self.precision[i].result() for i in range(self.n_label)])
        recall =tf.stack([self.recall[i].result() for i in range(self.n_label)])
        #precision = self.precision.result()
        #recall = self.recall.result()
        score = (1 + self.beta**2) * precision * recall / (self.beta**2 * precision + recall + self.epsilon)
        return score

    def reset_state(self):
        for i in range(self.n_label):
            self.precision[i].reset_state()
            self.recall[i].reset_state()
        return

    def get_config(self):
        """Returns the serializable config of the metric."""
        config = super().get_config()
        config.update({
            'beta': self.beta,
            'epsilon': self.epsilon,
            'thresholds': self.thresholds,
            'resize_factor': self.resize_factor,
        })
        return config

tf.keras.utils.get_custom_objects().update(FBetaScore=FBetaScore)
