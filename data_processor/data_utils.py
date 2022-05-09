import ast
import codecs
import json

import seaborn as sns
import matplotlib.pyplot as plt


def str2list(string):
    return ast.literal_eval(string)


def list2str(lst):
    return ' '.join(lst)


def write_json_file(file_name, data):
    with codecs.open(file_name, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False)
    print('Wrote {} records to {}'.format(len(data), file_name))


def dump_jsonl(data, output_path, append=False):
    """
    Write list of objects to a JSON lines file.
    """
    mode = 'a+' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + '\n')
    print('Wrote {} records to {}'.format(len(data), output_path))


def parse_data(file_name):
    data = {}

    with open(file_name, 'r') as f:
        doc_id = ''
        for line in f:
            if len(line) <= 2:
                continue
            if '#doc' in line:
                doc_id = line.strip()
                data[doc_id] = {}
                data[doc_id]['tokens'] = []
                data[doc_id]['relations'] = []
                data[doc_id]['heads'] = []
                data[doc_id]['BIO0s'] = []
                data[doc_id]['BIO1s'] = []
                data[doc_id]['BIO2s'] = []
            else:
                info = line.strip().split('\t')

                data[doc_id]['tokens'].append(info[1])
                data[doc_id]['relations'].append(str2list(info[3]))
                data[doc_id]['heads'].append(str2list(info[4]))

                data[doc_id]['BIO0s'].append(info[5])
                data[doc_id]['BIO1s'].append(info[6])
                data[doc_id]['BIO2s'].append(info[7])

    return data


def get_ner_labels(dataset):
    ner_label = []

    for lst in dataset['entities']:
        for sub_list in lst:
            for ele in sub_list:
                if type(ele) == str:
                    ner_label.append(ele)

    ner_labels = list(set(ner_label))
    return ner_labels


def get_relation_labels(dataset):
    rel_label = []

    for lst in dataset['relations']:
        for sub_list in lst:
            for ele in sub_list:
                if type(ele) == str:
                    rel_label.append(ele)

    rel_labels = list(set(rel_label))
    return rel_labels


def get_boundaries(bio):
    """
    Extracts an ordered list of boundaries. BIO label sequences can be either
    -     Raw BIO: B     I     I     O => {(0, 2, None)}
    - Labeled BIO: B-PER I-PER B-LOC O => {(0, 1, "PER"), (2, 2, "LOC")}
    """
    boundaries = []
    i = 0

    while i < len(bio):
        if bio[i][0] == 'O':
            i += 1
        else:
            s = i
            entity = bio[s][2:] if len(bio[s]) > 2 else None
            i += 1
            while i < len(bio) and bio[i][0] == 'I':
                if len(bio[i]) > 2 and bio[i][2:] != entity: break
                i += 1
            # boundaries.append([s, i - 1, entity])
            boundaries.append([s, i, entity])

    return boundaries


def get_relation_head_pairs(rel_value, rel_index):
    """
    Extracts an ordered list of relation-index-pairs.
    - idx: 0    relation_labels: ['Hypernym', 'Hypernym']	relation_index: [3, 3]
    - => [(0, 'Hypernym', 3)] which maps [('s_start_index', 'relation_value', 'o_start_index')]
    """
    relations = []

    for i, (idxs, labels) in enumerate(zip(rel_index, rel_value)):
        for idx, label in zip(idxs, labels):
            if label == 'N':
                continue
            else:
                # relations.append([i, idx, label])
                relations.append([idx, i, label])

    return relations


def ner_list2dict(lst):
    nested_ner_list = []
    for ele in range(len(lst)):
        dic = {'label': lst[ele][2], "start": lst[ele][0], "end": lst[ele][1]}
        nested_ner_list.append(dic)

    return nested_ner_list


def rel_list2dict(lst):
    nested_rel_list = []
    for ele in range(len(lst)):
        if lst:
            dic = {'label': lst[ele][2], 'sub': lst[ele][0], 'obj': lst[ele][1]}
        nested_rel_list.append(dic)

    return nested_rel_list


def plot_seq_len(dataset):
    sns.set_style('darkgrid')
    plt.figure(figsize=(16, 10))

    seq_len = []
    for data in dataset:
        seq_len.append(len(data['tokens']))
    sns.displot(seq_len)
    plt.show()

