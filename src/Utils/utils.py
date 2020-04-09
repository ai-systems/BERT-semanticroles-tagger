import pickle
from itertools import islice
from Utils.Features import *
import numpy as np


def load_pickle_file(filename):
    with open(filename, 'rb') as handle:
        file = pickle.load(handle)
    return file


def save_dict_as_pkl(dict_file, path):
    with open(path, 'wb') as handle:
        pickle.dump(dict_file, handle, protocol=pickle.HIGHEST_PROTOCOL)


def build_tag_vocab(labels):
    # Build dictionary with tag vocabulary
    tags = set([])

    tag2int = {}
    int2tag = {}

    for ts in labels:
        for t in ts:
            tags.add(t)

    for i, tag in enumerate(sorted(tags)):
        tag2int[tag] = i + 1
        int2tag[i + 1] = tag

    # Special character for the tags
    tag2int['-PAD-'] = 0
    int2tag[0] = '-PAD-'

    return tag2int, int2tag


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def split_sentence(sentences, max_len):
    new = []
    for data_t in sentences:
        data = data_t[1].strip().split(" ")
        term = data_t[0].strip()
        abc = []
        max_30 = list(chunk(data, max_len))
        for i in max_30:
            abc.append([(term), i])
        new.append(abc)

    new = [val for sublist in new for val in sublist]
    return new


def make_prediction_test(i, tokenizer, c_input_ids, yc_pred, int2tag):
    predictions = []
    for w, pred in zip(c_input_ids[i], yc_pred[i]):
        if tokenizer.convert_ids_to_tokens([w])[0] != '[PAD]' and \
                tokenizer.convert_ids_to_tokens([w])[0] != '[CLS]' and \
                tokenizer.convert_ids_to_tokens([w])[0] != '[SEP]':
            t = (tokenizer.convert_ids_to_tokens([w])[0], int2tag[pred])
            predictions.append(t)
    return predictions


def format_output(mylist):
    final_list = []
    for ele in mylist:
        str_op = ''
        str_op = str_op + ele[0] + "::"
        ser_in = ele[1]
        ser_out = ''
        for i in range(0, len(ser_in) - 1):
            if ser_in[i][1] != ser_in[i + 1][1]:
                mystr = ser_in[i][0] + "|" + ser_in[i][1] + ';'
                ser_out = ser_out + ' ' + mystr
            elif ser_in[i][1] == ser_in[i + 1][1]:
                ser_out = ser_out + ' ' + ser_in[i][0]
        # Add the last element from the list
        ser_out = ser_out + ' ' + ser_in[-1][0] + "|" + ser_in[-1][1]
        str_op = str_op + ser_out
        final_list.append(str_op)
    return final_list


def clean_test_data(df, MAX_SEQUENCE_LENGTH):
    # remove unwanted definition whose lenght is less than 20 char.
    df = df[(df.Definition.str.len() > 20)]
    df['Definition'] = df["Definition"].str.lower()
    df['Definition'] = df["Definition"].str.replace(',', '')
    df['Definition'] = df["Definition"].str.replace('.', '')
    defs = df['Definition'].tolist()
    terms = df['Term'].tolist()
    combined = list(zip(terms, defs))
    temp = split_sentence(combined, MAX_SEQUENCE_LENGTH)
    terms_final = []
    defs_final = []
    for e in temp:
        terms_final.append(e[0])
        defs_final.append(list(e[1]))
    return defs_final, terms_final


def convert_data_to_features(defs_final, MAX_SEQUENCE_LENGTH, tag2int, tokenizer):
    covid_fake_labels = []
    for item in defs_final:
        covid_fake_labels.append(['-PAD-'] * len(item))
    covid_example = convert_text_to_examples(defs_final, covid_fake_labels)

    (c_input_ids, c_input_masks, c_segment_ids, c_temp
     ) = convert_examples_to_features(tag2int, tokenizer, covid_example, max_seq_length=MAX_SEQUENCE_LENGTH + 2)

    return c_input_ids, c_input_masks, c_segment_ids


def make_prediction_on_test(c_input_ids, c_input_masks, c_segment_ids, model, tokenizer, int2tag, terms_final):
    yc_pred = model.predict([c_input_ids, c_input_masks, c_segment_ids]).argmax(-1)

    predict = []
    for i in range(0, len(yc_pred)):
        term = terms_final[i]
        ser = make_prediction_test(i, tokenizer, c_input_ids, yc_pred, int2tag)
        predict.append([term, ser])
    return predict


def preprocess_train_data(data_file):
    # Loop the data lines
    with open(data_file, 'r', encoding="utf-8", errors='ignore') as temp_f:
        # Read the lines
        lines = temp_f.readlines()

    formatted_sent = []
    terms = []
    for line in lines:
        formatted = []
        line_list = line.replace('&nbsp;', '').replace('&shy;', '').replace('&ndash;', '').split(";")
        term = line_list[2]
        term_pos = line_list[1]
        for words in line_list[4:]:
            words = words.replace('\n', '').replace('&nbsp;', '')
            l = words.split("/")
            if len(l) > 1:
                if l[1] in ['O', 'SUPERTYPE', 'ORIGIN-LOCATION', 'DIFFERENTIA-QUALITY', 'QUALITY-MODIFIER',
                            'DIFFERENTIA-EVENT-PARTICLE', 'ACCESSORY-QUALITY',
                            'ACCESSORY-DETERMINER', 'PURPOSE', 'EVENT-TIME', 'DIFFERENTIA-EVENT', 'ASSOCIATED-FACT',
                            'EVENT-LOCATION']:
                    if len(l[0]) > 1:
                        k = l[0].split(" ")
                        try:
                            for ele in k:
                                formatted.append((ele, l[1]))
                        except IndexError:
                            print(k)
                    elif len(l[0]) == 1:
                        try:
                            formatted.append((l[0], l[1]))
                        except IndexError:
                            print(k)
        if len(formatted) >= 2:
            formatted_sent.append(formatted)
            terms.append((term, term_pos))

    tagged_sentences = formatted_sent
    sentences, sentence_tags = [], []
    for tagged_sentence in tagged_sentences:
        sentence, tags = zip(*tagged_sentence)
        sentences.append(np.array(sentence))
        sentence_tags.append(np.array(tags))

    return sentences, sentence_tags
