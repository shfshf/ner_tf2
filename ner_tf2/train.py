# !/usr/bin/env python
# -*- coding:utf-8 -*-


from utils import tokenize, build_vocab, read_vocab
import tensorflow as tf
import tensorflow_addons as tf_ad
import os
import numpy as np
from args_help import args
from my_log import logger


if not (os.path.exists(args.vocab_file) and os.path.exists(args.tag_file)):
    logger.info("building vocab file")
    build_vocab([args.train_path], args.vocab_file, args.tag_file)
else:
    logger.info("vocab file exits!!")


vocab2id, id2vocab = read_vocab(args.vocab_file)
tag2id, id2tag = read_vocab(args.tag_file)
text_sequences_train, label_sequences_train = tokenize(args.train_path, vocab2id, tag2id)
text_sequences_test, label_sequences_test = tokenize(args.test_path, vocab2id, tag2id)

# train
train_dataset = tf.data.Dataset.from_tensor_slices((text_sequences_train, label_sequences_train))
train_dataset = train_dataset.shuffle(len(text_sequences_train)).batch(args.batch_size, drop_remainder=True)

# test
test_dataset = tf.data.Dataset.from_tensor_slices((text_sequences_test, label_sequences_test))
test_dataset = test_dataset.shuffle(len(text_sequences_test)).batch(args.batch_size, drop_remainder=True)

logger.info("hidden_num:{}, vocab_size:{}, label_size:{}".format(args.hidden_num, len(vocab2id), len(tag2id)))



