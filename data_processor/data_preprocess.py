import json
import os
import random

from data_utils import *

raw_data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
semreldata_file = raw_data_root + '/semreldata.tsv'

semreldata = parse_data(semreldata_file)

for keys, values in semreldata.items():
    entities = []
    for key, value in values.items():
        if key == 'BIO0s':
            span_BIO0s = get_boundaries(value)
            entities += span_BIO0s
        if key == 'BIO1s':
            span_BIO1s = get_boundaries(value)
            entities += span_BIO1s
        if key == 'BIO2s':
            span_BIO2s = get_boundaries(value)
            entities += span_BIO2s

    semreldata[keys]['entities'] = entities

    del semreldata[keys]['BIO0s']
    del semreldata[keys]['BIO1s']
    del semreldata[keys]['BIO2s']

    relation_head_pair = get_relation_head_pairs(semreldata[keys]['relations'], semreldata[keys]['heads'])
    semreldata[keys]['relation_head'] = relation_head_pair

    # span_relation_pair = get_span_relation(semreldata[keys]['entities'], semreldata[keys]['relation_head'])
    # semreldata[keys]['span_relation'] = span_relation_pair

    del semreldata[keys]['relations']
    del semreldata[keys]['heads']
    # del semreldata[keys]['relation_head']

span_data = []
for key, value in semreldata.items():
    single_dict = {
        'doc_key': key,
        'tokens': semreldata[key]['tokens'],
        'entities': ner_list2dict(semreldata[key]['entities']),
        'relations': rel_list2dict(semreldata[key]['relation_head'])
    }
    span_data.append(single_dict)

# randomly split the train - dev : 80% - 20% dataset
random.shuffle(span_data)
train = span_data[:int(len(span_data) * 0.8)]
dev = span_data[int(len(span_data) * 0.8):]

# write_json_file(raw_data_root+'/train.json', train)
# write_json_file(raw_data_root+'/dev.json', dev)

# train_data = json.load(open(raw_data_root + '/train.json'))
# dev_data = json.load(open(raw_data_root + '/dev.json'))



