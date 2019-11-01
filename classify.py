import argparse
import pandas as pd
import pickle
import numpy as np


from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer

from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=10, inter_op_parallelism_threads=10)))


import re
import requests
import mwparserfromhell


API_URL = "https://en.wikipedia.org/w/api.php"


'''
This function takes a text and splits it into sentences
'''


def make_sentence(text):
    text = text.replace("\n", "")
    sentences = text.split(".")
    return sentences


'''
This function takes the title of an article and creates an input.tsv file of sentences and their sections
to be classified by the citation needed model
'''


def parse():
    title = raw_input('Title of wikipedia article?')
    # retrieves text of article from wikimedia
    params = {'action': 'query',
              'format': 'json',
              'titles': title,
              'prop': 'extracts',
              'explaintext': True}
    headers = {"User-Agent": "My-Bot-Name/1.0"}
    response = requests.get(API_URL, headers=headers, params=params).json()
    page = next(iter(response['query']['pages'].values()))
    text = mwparserfromhell.parse(page['extract'])
    # sections to be ignored
    exempt_sections = ['See also', 'External links', 'References', 'Further reading', 'Notes', 'Other print references',
                       'Historically important texts', 'Textbooks and general references']
    sections = text.get_sections(flat=True, include_lead=True, include_headings=True, levels=[2])
    statements = {}
    # stores each valid section and section content as key value pair dictionary statement
    for section in sections:
        if not section.filter_headings():
            section_name = 'Main_section'
            section_content = section.strip_code()
            sentences = make_sentence(section_content)
            statements[section_name] = sentences
        elif section.filter_headings()[0].title in exempt_sections:
            continue
        else:
            section_name = str(section.filter_headings()[0].title)
            section_content = section.strip_code()
            sentences = make_sentence(section_content)
            statements[section_name] = sentences
    # writes each sentence and it's section in a file to be processed by the citation needed model
    input_string = 'section\tstatement\n'
    for section in statements:
        section = section.encode('UTF-8')
        for sentence in statements[section]:
            sentence = sentence.encode('UTF-8')
            input_string += section + '\t' + sentence + '\t' + '\n'
    input_file = open('input.tsv', 'wt')
    input_file.write(input_string)
    input_file.flush()
    input_file.close()


'''
    Set up the arguments and parse them.
'''


def get_arguments():
    parser = argparse.ArgumentParser(
        description='Use this script to determinee whether a statement needs a citation or not.')
    # parser.add_argument('-i', '--input', help='The input file from which we read the statements. Lines contains tab-separated values: the statement, the section header, and additionally the binary label corresponding to whether the sentence has a citation or not in the original text. This can be set to 0 if no evaluation is needed.', required=True)
    parser.add_argument('-o', '--out_dir', help='The output directory where we store the results', required=True)
    parser.add_argument('-m', '--model', help='The path to the model which we use for classifying the statements.', required=True)
    parser.add_argument('-v', '--vocab', help='The path to the vocabulary of words we use to represent the statements.', required=True)
    parser.add_argument('-s', '--sections', help='The path to the vocabulary of section with which we trained our model.', required=True)
    parser.add_argument('-l', '--lang', help='The language that we are parsing now.', required=False, default='en')

    return parser.parse_args()


'''
    Parse and construct the word representation for a sentence.
'''


def text_to_word_list(text):
    # check first if the statements is longer than a single sentence.
    sentences = re.compile('\.\s+').split(str(text))
    if len(sentences) != 1:
        # text = sentences[random.randint(0, len(sentences) - 1)]
        text = sentences[0]

    text = str(text).lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.strip().split()

    return text


'''
    Compute P/R/F1 from the confusion matrix.
'''


'''
    Create the instances from our datasets
'''


def construct_instance_reasons(statement_path, section_dict_path, vocab_w2v_path, max_len=-1):
    # Load the vocabulary
    vocab_w2v = pickle.load(open(vocab_w2v_path, 'rb'))

    # load the section dictionary.
    section_dict = pickle.load(open(section_dict_path, 'rb'))

    # Load the statements, the first column is the statement and the second is the label (True or False)
    statements = pd.read_csv(statement_path, sep='\t', index_col=False, error_bad_lines=False, warn_bad_lines=False)
    # removes all rows with NAN values
    statements = statements.dropna()

    # construct the training data
    X = []
    sections = []
    outstring=[]

    for index, row in statements.iterrows():
        try:
            statement_text = text_to_word_list(row['statement'])

            X_inst = []
            for word in statement_text:
                if max_len != -1 and len(X_inst) >= max_len:
                    continue
                if word not in vocab_w2v:
                    X_inst.append(vocab_w2v['UNK'])
                else:
                    X_inst.append(vocab_w2v[word])

            # extract the section, and in case the section does not exist in the model, then assign UNK
            section = row['section'].strip().lower()
            sections.append(np.array([section_dict[section] if section in section_dict else 0]))

            X.append(X_inst)
            outstring.append(str(row["statement"]))

        except Exception as e:
            print row
            print e.message
    X = pad_sequences(X, maxlen=max_len, value=vocab_w2v['UNK'], padding='pre')

    encoder = LabelBinarizer()


    return X, np.array(sections),  encoder, outstring


if __name__ == '__main__':
    parse()
    p = get_arguments()

    # load the model
    model = load_model(p.model)

    # load the data
    max_seq_length = model.input[0].shape[1].value
    X, sections, encoder, outstring = construct_instance_reasons('input.tsv', p.sections, p.vocab, max_seq_length)

    # classify the data
    pred = model.predict([X, sections])

    # sort by predicted score
    pred_df = pd.DataFrame(pred).sort_values(by=[0])

    outstr = 'Text\tPrediction\n'
    # save the prediction
    for idx, y_pred in pred_df.iterrows():
        outstr += outstring[idx] + '\t' + str(y_pred[0]) + '\t' + '\n'

    fout = open(p.out_dir + '/' + p.lang + '_predictions_sections.tsv', 'wt')
    fout.write(outstr)
    fout.flush()
    fout.close()
