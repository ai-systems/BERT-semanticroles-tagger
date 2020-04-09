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

# Build model
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
    if mode == 'train':
        run_train(config)
    if mode == 'eval':
        run_eval(config)


def run_train(config):
    print("in train")
     # sentences,labels = preprocess_train_data(config['path']['train_data'])
    sentences = load_pickle_file(config['path']['sentences'])
    labels = load_pickle_file(config['path']['labels'])

    # split the data in train and test set
    (train_text, test_text, train_labels, test_labels) = train_test_split(sentences, labels, test_size=0.1)
    tag2int, int2tag = build_tag_vocab(labels)

    #save tags for testing
    # save_dict_as_pkl(tag2int,config['path']['tag2int'])
    # save_dict_as_pkl(int2tag, config['path']['int2tag'])

    # filtering out the sentences with less than 30 , could split the data as well.
    MAX_SEQUENCE_LENGTH = config['common']['max_seq_len']
    test_text = [item for item in test_text if len(item) <= MAX_SEQUENCE_LENGTH]
    train_text = [item for item in train_text if len(item) <= MAX_SEQUENCE_LENGTH]
    test_labels = [item for item in test_labels if len(item) <= MAX_SEQUENCE_LENGTH]
    train_labels = [item for item in train_labels if len(item) <= MAX_SEQUENCE_LENGTH]

    # Initialize session
    sess = tf.Session()
    # Params for bert model and tokenization
    bert_path = config['path']['bert_path']

    tokenizer = create_tokenizer_from_hub_module(bert_path, sess)

    # Convert data to format
    train_examples = convert_text_to_examples(train_text, train_labels)
    test_examples = convert_text_to_examples(test_text, test_labels)

    # # Convert to features
    (train_input_ids, train_input_masks, train_segment_ids, train_labels_ids  # train_labels
     ) = convert_examples_to_features(tag2int, tokenizer, train_examples, max_seq_length=MAX_SEQUENCE_LENGTH + 2)
    (test_input_ids, test_input_masks, test_segment_ids, test_labels_ids
     ) = convert_examples_to_features(tag2int, tokenizer, test_examples, max_seq_length=MAX_SEQUENCE_LENGTH + 2)

    # one hot encoding
    n_tags = len(tag2int)
    train_labels = to_categorical(train_labels_ids, num_classes=n_tags)
    test_labels = to_categorical(test_labels_ids, num_classes=n_tags)

    model = build_model(MAX_SEQUENCE_LENGTH + 2, n_tags)

    EPOCHS = config['hyperparams']['epochs']

    t_ini = datetime.datetime.now()
    filepath = config['path']['model_checkpoint'] + 'weights-improvement-{epoch:02d}-val_acc-{val_accuracy:.3f}.hdf5'
    cp = ModelCheckpoint(filepath=filepath,
                         monitor='val_accuracy',
                         save_best_only=True,
                         save_weights_only=True,
                         verbose=1)

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=2)

    history = model.fit([train_input_ids, train_input_masks, train_segment_ids],
                        train_labels,
                        validation_data=([test_input_ids, test_input_masks, test_segment_ids], test_labels),
                        # validation_split=0.2,
                        epochs=EPOCHS,
                        batch_size=config['hyperparams']['batch_size'],
                        shuffle=True,
                        verbose=1,
                        callbacks=[cp, early_stopping]
                        )

    t_fin = datetime.datetime.now()
    print('Training completed in {} seconds'.format((t_fin - t_ini).total_seconds()))
    y_pred = model.predict([test_input_ids, test_input_masks, test_segment_ids]).argmax(-1)
    y_true = test_labels.argmax(-1)
    y_zipped = zip(y_true.flat, y_pred.flat)
    y_true, y_pred = y2label(int2tag, y_zipped)
    name = 'Bert fine-tuned model'
    print('\n------------ Result of {} ----------\n'.format(name))
    print(classification_report(y_true, y_pred, digits=4))

    print("Accuracy: {0:.4f}".format(accuracy_score(y_true, y_pred)))
    print('f1-macro score: {0:.4f}'.format(f1_score(y_true, y_pred, average='macro')))


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
    raw = pd.read_csv(config['path']['test_data'])
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

    with open('.\config\config.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    run_process(mode,config)
