from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, TensorBoard
from keras import backend as K
import datetime
from sklearn.metrics import recall_score, precision_score, classification_report, accuracy_score, confusion_matrix, \
    f1_score
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import tensorflow as tf
import pandas as pd
import keras
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
import sys
import getopt

from Utils.utils import *
from Utils.BERT import *
from Utils.Features import *


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


def y2label(int2tag, zipped, mask=0):
    out_true = []
    out_pred = []
    for zip_i in zipped:
        a, b = tuple(zip_i)
        if a != mask:
            out_true.append(int2tag[a])
            out_pred.append(int2tag[b])
    return out_true, out_pred


def run_process(mode,config):
    # if mode == 'train':
    #     run_train(config)
    if mode == 'eval':
        run_eval(config)

def preprocess_data(path):
#     logger.info("preprocessing data from ::" + path)
    with open(path, 'r') as infile:
        data = json.load(infile)
    data_set = []

    for row in data['questions']:
        temp = row['snippets']
        for item in temp:
            temp_set = {}
            # temp_set['questions'] = row['body']
            temp_set['Definition'] = item['text']
            # temp_set['type'] = row['type']
            temp_set['Term'] = row['id']
            # try:
            #     temp_set['exact_answer'] = row['exact_answer']
            # except:
            #     temp_set['exact_answer'] = 'NA'

                # logger.warning("Missing exact answer!! Replacing with NA")
            data_set.append(temp_set)

    df = pd.DataFrame(data_set)
    return df

def build_model(max_seq_length, n_tags):
    seed = 0
    in_id = keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    np.random.seed(seed)
    bert_output = BertLayer()(bert_inputs)

    np.random.seed(seed)
    outputs = keras.layers.Dense(n_tags, activation=keras.activations.softmax)(bert_output)

    np.random.seed(seed)
    model = keras.models.Model(inputs=bert_inputs, outputs=outputs)
    np.random.seed(seed)
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00005), loss=keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary(100)
    return model

def run_eval(config):
    print("in eval")
    # load tags
    int2tag = load_pickle_file(config['path']['int2tag'])
    tag2int = load_pickle_file(config['path']['tag2int'])

    n_tags = len(tag2int)
    # load model
    MAX_SEQUENCE_LENGTH = config['common']['max_seq_len']
    model = build_model(MAX_SEQUENCE_LENGTH + 2, n_tags)
    model.load_weights(config['path']['default_model_for_test'])

    # load tf session
    sess = tf.Session()
    # Params for bert model and tokenization
    bert_path = config['path']['bert_path']
    tokenizer = create_tokenizer_from_hub_module(bert_path, sess)

    # load definition dataset
    # raw_df = pd.read_csv(config['path']['bioasq_data'])
    raw = preprocess_data(config['path']['bioasq_data'])
    # raw = raw[:10]

    df = raw[['Term', 'Definition']]

    # preprocess
    defs_final, terms_final = clean_test_data(df,MAX_SEQUENCE_LENGTH)

    # convert_data_to_features
    c_input_ids, c_input_masks, c_segment_ids = convert_data_to_features(defs_final, MAX_SEQUENCE_LENGTH, tag2int,tokenizer)

    # get the prediction
    predicted_op = make_prediction_on_test(c_input_ids, c_input_masks, c_segment_ids, model, tokenizer, int2tag,
                                           terms_final)
    # format the output to get the contiguous sentences
    out = format_output(predicted_op)
    out_df = pd.DataFrame.from_records([sub.split("::") for sub in out], columns=['term', 'ser'])
    out_df = out_df.groupby('term').agg({'ser': 'sum'})
    out_df.to_csv(config['path']['ser_tagged_data'])


if __name__ == "__main__":
    try:
        options, args = getopt.getopt(sys.argv[1:], "mh", ["mode="])
        for name, value in options:
            if name in ('-m', '--mode'):
                mode = value
                assert mode == "train" or mode == "eval"
            if name in ('-h', '--help'):
                print ('python ser_bert_tagger.py --mode eval\\train ')
                sys.exit(1)
    except getopt.GetoptError as err:
        print("Seems arguments are wrong..")
        print("usage:: python ser_bert_tagger.py --mode eval\\train")
        print ("Ex:: python ser_bert_tagger.py --mode eval")
        sys.exit(1)

    with open('./config/config_bioasq.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile)
    run_process(mode,config)
