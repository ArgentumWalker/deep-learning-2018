import tensorflow as tf
import os
from pandas import DataFrame
from tensorboard.plugins import projector

from utils.utils import acid_dict, save_tsv


def save_embedding(directory, dictionary, data, props):
    path = os.getcwd()
    log_dir = path + '/' + directory

    dicts = [acid_dict(codone, props) for codone in dictionary.keys()]
    columns_names = set()
    for d in dicts:
        columns_names.update(d.keys())

    labels = DataFrame(columns=list(columns_names))
    sorted_data = []

    for codone in dictionary.keys():
        codone_props = acid_dict(codone, props)
        labels = labels.append(codone_props, ignore_index=True)
        sorted_data.append(list(data[dictionary[codone]]))

    save_tsv(labels, log_dir, "labels")

    tf_data = tf.Variable(sorted_data)
    with tf.Session() as sess:
        saver = tf.train.Saver([tf_data])
        sess.run(tf_data.initializer)
        saver.save(sess, os.path.join(log_dir, 'data.ckpt'))
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = tf_data.name
        embedding.metadata_path = os.path.join(log_dir, 'labels.tsv')
        projector.visualize_embeddings(tf.summary.FileWriter(log_dir), config)