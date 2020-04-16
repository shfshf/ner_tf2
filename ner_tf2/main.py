import os
import tensorflow as tf

from ioflow.configure import read_configure
from ioflow.corpus import get_corpus_processor
from tokenizer_tools.tagset.converter.offset_to_biluo import offset_to_biluo
from ner_tf2.input import generate_tagset, Lookuper, index_table_from_file


def main():
    config = read_configure()  # ioflow
    corpus = get_corpus_processor(config)
    corpus.prepare()  # ?
    train_data_generator_func = corpus.get_generator_func(corpus.TRAIN)
    eval_data_generator_func = corpus.get_generator_func(corpus.EVAL)

    corpus_meta_data = corpus.get_meta_info()
    tags_data = generate_tagset(corpus_meta_data["tags"])  # process entity into BIO

    train_data = list(train_data_generator_func())
    eval_data = list(eval_data_generator_func())

    tag_lookuper = Lookuper({v: i for i, v in enumerate(tags_data)})  # tag index
    vocab_data_file = config.get("vocabulary_file")

    if not vocab_data_file:
        # load built in vocabulary file
        vocab_data_file = os.path.join(
            os.path.dirname(__file__), "../data/unicode_char_list.txt"
        )

    vocabulary_lookuper = index_table_from_file(vocab_data_file)

    def preprocss(data, maxlen):
        raw_x = []
        raw_y = []

        for offset_data in data:
            tags = offset_to_biluo(offset_data)
            words = offset_data.text

            tag_ids = [tag_lookuper.lookup(i) for i in tags]
            word_ids = [vocabulary_lookuper.lookup(i) for i in words]

            raw_x.append(word_ids)
            raw_y.append(tag_ids)

        if maxlen is None:
            maxlen = max(len(s) for s in raw_x)

        print(">>> maxlen: {}".format(maxlen))

        x = tf.keras.preprocessing.sequence.pad_sequences(
            raw_x, maxlen, padding="post"
        )  # right padding

        # lef padded with -1. Indeed, any integer works as it will be masked
        # y_pos = pad_sequences(y_pos, maxlen, value=-1)
        # y_chunk = pad_sequences(y_chunk, maxlen, value=-1)
        y = tf.keras.preprocessing.sequence.pad_sequences(
            raw_y, maxlen, value=0, padding="post"
        )

        return x, y

    MAX_SENTENCE_LEN = config.get("max_sentence_len", 25)

    train_x, train_y = preprocss(train_data, MAX_SENTENCE_LEN)
    test_x, test_y = preprocss(eval_data, MAX_SENTENCE_LEN)
    # print(test_x)
    # print(test_y)

    EPOCHS = config["epochs"]
    BATCH_SIZE = config["batch_size"]
    EMBED_DIM = config["embedding_dim"]
    USE_ATTENTION_LAYER = config.get("use_attention_layer", False)
    BiLSTM_STACK_CONFIG = config.get("bilstm_stack_config", [])
    BATCH_NORMALIZATION_AFTER_EMBEDDING_CONFIG = config.get(
        "use_batch_normalization_after_embedding", False
    )
    BATCH_NORMALIZATION_AFTER_BILSTM_CONFIG = config.get(
        "use_batch_normalization_after_bilstm", False
    )
    CRF_PARAMS = config.get("crf_params", {})
    OPTIMIZER_PARAMS = config.get("optimizer_params_keras", {})

    vacab_size = vocabulary_lookuper.size()
    tag_size = tag_lookuper.size()




if __name__ == "__main__":
    main()