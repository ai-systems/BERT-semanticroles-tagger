import scispacy
import spacy
import time
import pandas as pd
from scispacy.abbreviation import AbbreviationDetector
from scispacy.umls_linking import UmlsEntityLinker

linker = UmlsEntityLinker(resolve_abbreviations=True)


def generate_defs_corpus(config):
    """
    This scripts reads the `abstract` from the corpus(csv) and extract the definition from sciSpacy for all
    the keywords and saves these definition in a text file.
    Since, the extraction is slow process , we are writing the definition in a text file for every paragraph
    then run `create_definition_file.py` to generate the final csv file with `term` and `definition`

    """
    raw = pd.read_csv(config['path']['corpus'])
    raw = raw[['abstract']]
    raw['abstract'] = raw['abstract'].str.lower()
    raw.dropna(inplace=True)
    abstract_list = raw['abstract'].tolist()
    nlp = spacy.load("en_core_sci_sm")
    nlp.add_pipe(linker)
    print('total paragraphs:::', len(abstract_list))
    count = 0
    # for each paragraph extarct all definition and write the definitions to a temp location
    for text in abstract_list:
        doc = nlp(text)
        # time.sleep(1)
        spacy_out = []
        for entity in doc.ents:
            for umls_ent in entity._.umls_ents:
                spacy_out.append(linker.umls.cui_to_entity[umls_ent[0]])
        name = config['path']['temp_dir'] + str(c)
        if len(spacy_out) > 10:
            fo = open('{0}.txt'.format(name), 'w', encoding="utf-8")
            for ele in spacy_out:
                fo.write(str(ele) + '\n')
            fo.close()
        print('writen::::' + str(c))
        count = count + 1


if __name__ == "__main__":
    with open('.\config\config_defs.yaml', 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    generate_defs_corpus(config)
