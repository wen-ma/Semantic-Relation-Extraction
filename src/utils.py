import os
import copy
import json
import random
import logging
import warnings
import argparse
import transformers
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_cosine_schedule_with_warmup

from models import *

# Set system log at level error to reduce the useless log interference
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

# Define parameters
parser = argparse.ArgumentParser()
parser.add_argument("--root_dir", default="/Users/TheaMA/PycharmProjects/Relation-Extraction", type=str)
parser.add_argument("--dataset", default="CoNLL04", type=str)
parser.add_argument("--saved_model", default="CoNLL04", type=str)
parser.add_argument("--result_file_name", default="CoNLL04", type=str)
parser.add_argument("--log_name", default="CoNLL04", type=str)
parser.add_argument("--epoch", default=10, type=int)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--batch_print", default=32, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--max_length", default=512, type=int)
parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
parser.add_argument("--do_data_prepare", action="store_true")
parser.add_argument("--do_eval", action="store_true")
args = parser.parse_args()

# Go to the current path
os.chdir(args.root_dir)


def get_logger(filename, verbosity=1, name=None):
    """Define logger configuration to display on the console and log to file"""
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


log_file = get_logger(filename=f"./logs/{args.log_name}.log")
log_file.info(f"-------------------------------------------new_log------------------------------------------")
log_file.info(f"input args:{args}")
log_file.info(f"executive dir: {os.getcwd()}")

"""
Function to read and write data
"""
def read_json(file): return json.load(open(file))
def read_jsonl(file): return [json.loads(line) for line in open(file).readlines()]
def write_json(obj, file): json.dump(obj, open(file, "w"))
def w2f(file, str):
    with open(file, "a") as f:
        f.write(str + "\n")


"""
Data preparation
"""
if args.do_data_prepare:
    # Read the raw data
    data_path = f"./data/{args.dataset}/"
    log_file.info(f"handle {args.dataset} data...")
    train_path = data_path + "train" + ".json"
    dev_path = data_path + "dev" + ".json"
    train, dev = read_json(train_path), read_json(dev_path)
    # Create json file of ner and relation labels respectively if they do not exist.
    ner_label_path = data_path + "ner_label.json"
    rel_label_path = data_path + "rel_label.json"
    if not os.path.exists(rel_label_path):
        log_file.info("Creating ner and rel - label2id...")


        def get_ner_label(dic_list):
            ent_list = [j["label"] for i in dic_list for j in i["entities"]]
            return list(set(ent_list))


        def get_rel_label(dic_list):
            rel_list = [j["label"] for i in dic_list for j in i["relations"]]
            rel_list.append("no_relation")
            return list(set(rel_list))


        ner_label = get_ner_label(train)
        rel_label = get_rel_label(train)
        log_file.info(f"ner_label:{ner_label}, rel_label:{rel_label}")
        write_json({k: v for k, v in zip(ner_label, range(len(ner_label)))}, ner_label_path)
        write_json({k: v for k, v in zip(rel_label, range(len(rel_label)))}, rel_label_path)

    # Add marked type info into train data
    dname = args.root_dir + f"/processed_data/{args.dataset}/"
    if not os.path.isdir(dname):
        log_file.info("prepare training data...")


        def add_train_track(dic):
            dic["train_data"] = []
            start2label = {i["start"]: (i["label"], i["end"]) for i in dic["entities"]}
            tmp = []
            for i in dic["relations"]:
                if i["label"] == "Synonym" and args.dataset == "SemRel_sym":
                    tmp.append({"label": "Synonym", "subj": i["obj"], "obj": i["subj"]})
                # use semantic relations symmetry to increase data points
                elif i["label"] == "Co-Hyponym" and args.dataset == "SemRel_sym":
                    tmp.append({"label": "Co-Hyponym", "subj": i["obj"], "obj": i["subj"]})
                else:
                    tmp.append({"label": "no_relation", "subj": i["obj"], "obj": i["subj"]})
            dic["relations"] += tmp
            raw_rel = {i["subj"]: i["obj"] for i in dic["relations"]}
            tmp = []
            for i in dic["entities"]:
                for j in dic["entities"]:
                    if i == j: continue
                    if i["start"] in raw_rel.keys() and j["start"] == raw_rel[i["start"]]: continue
                    tmp.append({"label": "no_relation", "subj": i["start"], "obj": j["start"]})
            random.shuffle(tmp)
            dic["relations"] += tmp[:(len(dic["relations"]) // 2)]
            for spo in dic["relations"]:
                try:
                    tokens = copy.deepcopy(dic["tokens"])
                    if spo["subj"] < spo["obj"]:
                        # add marked type of subj
                        subj_type = start2label[spo["subj"]][0].upper()
                        subj_start_index = spo["subj"]
                        tokens.insert(subj_start_index, f"<SUBJ_START_{subj_type}>")
                        subj_end_index = start2label[spo["subj"]][1] + 2
                        tokens.insert(subj_end_index, f"<SUBJ_END_{subj_type}>")
                        # add marked type of obj
                        obj_type = start2label[spo["obj"]][0].upper()
                        obj_start_index = spo["obj"] + 2
                        tokens.insert(obj_start_index, f"<OBJ_START_{obj_type}>")
                        obj_end_index = start2label[spo["obj"]][1] + 4
                        tokens.insert(obj_end_index, f"<OBJ_END_{obj_type}>")
                    else:
                        # add marked type of subj
                        obj_type = start2label[spo["obj"]][0].upper()
                        obj_start_index = spo["obj"]
                        tokens.insert(obj_start_index, f"<OBJ_START_{obj_type}>")
                        obj_end_index = start2label[spo["obj"]][1] + 2
                        tokens.insert(obj_end_index, f"<OBJ_END_{obj_type}>")
                        # add marked type of obj
                        subj_type = start2label[spo["subj"]][0].upper()
                        subj_start_index = spo["subj"] + 2
                        tokens.insert(subj_start_index, f"<SUBJ_START_{subj_type}>")
                        subj_end_index = start2label[spo["subj"]][1] + 4
                        tokens.insert(subj_end_index, f"<SUBJ_END_{subj_type}>")
                    dic["train_data"].append({"typed_tokens": tokens, "rel": spo["label"], "s_start": subj_start_index,
                                              "s_end": subj_end_index, "o_start": obj_start_index,
                                              "o_end": obj_end_index})
                except:
                    log_file.info(f"parsing error in doc_key:{dic['doc_key']} and spo:{spo}")
            return dic


        os.makedirs(dname, exist_ok=True)


        # Cross validation
        def partition(ls, size):
            par = [ls[i:i + size] for i in range(0, len(ls), size)]
            if len(par[-1]) != size:
                par[-2].extend(par.pop())
            return par


        train_split_fold = partition(train, int(len(train) // 10))
        assert len(train_split_fold) == 10
        for index, fold in enumerate(train_split_fold):
            for dic in fold:
                if len(dic["tokens"]) > 500 or dic["relations"] == []: continue
                dic = add_train_track(dic)
                w2f(dname + f"train_fold_{index}.json", json.dumps(dic))
        for i in dev:
            if len(i["tokens"]) > 500 or i["relations"] == []: continue
            dic = add_train_track(i)
            w2f(dname + "dev_all.json", json.dumps(dic))

    log_file.info(
        f"{args.dataset} data prepared, train_data length: {len(train)}, dev_data length: {len(dev)}, train_dir: {dname}")

# 10 fold cross validation
dname = args.root_dir + f"/processed_data/{args.dataset}/"
train_dir = [dname + f"train_fold_{i}.json" for i in range(10)]
assert len(train_dir) == 10
test_dir = dname + "dev_all.json"

ner_label2id = read_json(f"./data/{args.dataset}/ner_label.json")
rel_label2id = read_json(f"./data/{args.dataset}/rel_label.json")


def make_marker_tokens(ner_labels):
    new_tokens = []
    for label in ner_labels:
        new_tokens.append("<SUBJ_START_%s>" % label.upper())
        new_tokens.append("<SUBJ_END_%s>" % label.upper())
        new_tokens.append("<OBJ_START_%s>" % label.upper())
        new_tokens.append("<OBJ_END_%s>" % label.upper())
    return new_tokens


# Take marked type of sub and obj as special tokens and add them into tokenizer
tokenizer_kwargs = {
    "use_fast": True,
    "additional_special_tokens": make_marker_tokens(ner_label2id.keys()),
}
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", **tokenizer_kwargs)

"""
Define data loader
"""


class InputFeatures(Dataset):
    def __init__(self, data_list, rel_dir, tokenizer, maxlen, with_labels=True, model_name="bert-base-uncased"):
        self.data_list = data_list
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        self.with_labels = with_labels
        self.rel_label2id = read_json(rel_dir)

    def __len__(self):
        return len(self.data_list)

    def convert_examples_to_features(self):
        out = self.tokenizer(" ".join(self.dic["typed_tokens"]),
                             max_length=self.maxlen,
                             padding="max_length",
                             truncation=True,
                             add_special_tokens=True,
                             return_tensors="pt")
        t2i = tokenizer.convert_ids_to_tokens(out["input_ids"].squeeze().tolist())
        sub_idx = [t2i.index(i) for i in t2i if "<SUBJ_START_" in i]
        assert len(sub_idx) == 1
        obj_idx = [t2i.index(i) for i in t2i if "<OBJ_START_" in i]
        assert len(obj_idx) == 1
        sub_idx, obj_idx = sub_idx[0], obj_idx[0]
        return out, sub_idx, obj_idx

    def __getitem__(self, index):
        self.dic = self.data_list[index]
        out, sub_idx, obj_idx = self.convert_examples_to_features()
        input_ids = out["input_ids"].squeeze(0)
        attention_mask = out["attention_mask"].squeeze(0)
        token_type_ids = out["token_type_ids"].squeeze(0)
        if self.with_labels:  # True if the dataset has labels
            label = torch.tensor(int(self.rel_label2id[self.data_list[index]["rel"]])).squeeze(0)
            return input_ids, attention_mask, token_type_ids, label, torch.tensor(sub_idx), torch.tensor(obj_idx)
        else:
            return input_ids, attention_mask, token_type_ids, torch.tensor(sub_idx), torch.tensor(obj_idx)


# Parameters of BERT model
config = {
    "model_name_or_path": "bert-base-uncased",
    "hidden_dropout_prob": 0.2,
    "hidden_size": 768,
}

"""
Train data
"""


def save_trained_model(output_dir, model_name, model):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    log_file.info("Saving model to %s" % output_dir)
    model_to_save = model.module if hasattr(model, "module") else model
    torch.save(model_to_save.state_dict(), output_dir + "/" + model_name + ".pth")


# Start cross validation
for fold_round in range(10):
    model = RelationModel(config, num_rel_labels=len(rel_label2id))
    model.bert.resize_token_embeddings(len(tokenizer))
    train_data_list = []
    for index, tmp_dir in enumerate(train_dir):
        if index == fold_round:
            dev_data_list = [j for i in read_jsonl(tmp_dir) for j in i["train_data"]]
        else:
            tmp_data = [j for i in read_jsonl(tmp_dir) for j in i["train_data"]]
            train_data_list.extend(tmp_data)
    # train the model
    log_file.info(f"*****10 fold cross validation*****")
    log_file.info(f"*****fold_round:{fold_round} Running train *****")
    log_file.info(f"*****train_data_length:{len(train_data_list)}, dev_data_length:{len(dev_data_list)}*****")
    train_dataset = InputFeatures(data_list=train_data_list, rel_dir=f"./data/{args.dataset}/rel_label.json",
                                    tokenizer=tokenizer, maxlen=512, with_labels=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    if args.do_eval:
        dev_dataset = InputFeatures(data_list=dev_data_list, rel_dir=f"./data/{args.dataset}/rel_label.json",
                                      tokenizer=tokenizer, maxlen=512, with_labels=True)
        dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
        eval_acc_stack = []
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    epochs = args.epoch
    batch_print = args.batch_print
    best_loss = 100
    running_loss = 0.0
    count = 0
    loss_stack = []
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10,
                                                num_training_steps=total_steps)
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    for epoch in range(epochs):
        log_file.info(f"Epoch: {epoch}")
        for i,batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, token_type_ids, labels, sub_idx, obj_idx = batch
            loss = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels, sub_idx=sub_idx, obj_idx=obj_idx)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            if i % batch_print == 0 and i != 0:
                count += 1
                log_file.info(f"Epoch:{epoch}/{args.epoch},Batch:{i},Loss:{running_loss/batch_print}")
                loss_stack.append(running_loss/batch_print)
                running_loss = 0.0
        if args.do_eval:
            log_file.info(f"*****Epoch:{epoch}/{args.epoch} Running eval *****")
            right = 0
            total = 0
            for i,batch in tqdm(enumerate(dev_loader)):
                model.eval()
                with torch.no_grad():
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, attention_masks, token_type_ids, labels, sub_idx, obj_idx = batch
                    logits = model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, sub_idx=sub_idx, obj_idx=obj_idx)
                    if int(logits.argmax(dim=1)) == int(labels):
                        right += 1
                    total += 1
            eval_acc = right/total
            log_file.info(f"Eval Accuracy: {eval_acc}")
            if not eval_acc_stack:
                eval_acc_stack.append(eval_acc)
                save_trained_model(f"./models/{args.saved_model}", args.saved_model+f"_fold_{fold_round}", model)
                log_file.info(f"{args.saved_model}_fold_{fold_round} model saved")
            if eval_acc > max(eval_acc_stack):
                log_file.info(f"New Best eval accuracy: {eval_acc}, old best eval accuracy: {max(eval_acc_stack)}")
                save_trained_model(f"./models/{args.saved_model}", args.saved_model+f"_fold_{fold_round}", model)
                log_file.info(f"{args.saved_model}_fold_{fold_round} model saved")
                eval_acc_stack.append(eval_acc)
        else:
            if loss_stack[-1] < best_loss:
                best_loss = loss_stack[-1]
                log_file.info(f"New Best loss: {best_loss}, old best loss: {loss_stack[-1]}")
                save_trained_model(f"./models/{args.saved_model}", args.saved_model+f"_fold_{fold_round}", model)
                log_file.info(f"{args.saved_model}_fold_{fold_round} model saved")
                loss_stack = []
    # Test and record the results
    log_file.info(f"*****fold_round:{fold_round} Running Running test *****")
    model.module.load_state_dict(torch.load(f"./models/{args.saved_model}/"+args.saved_model+f"_fold_{fold_round}"+".pth")) # , map_location=torch.device(f"cuda:{args.cuda_num-1}"
    test_data = read_jsonl(f"./processed_data/{args.dataset}/dev_all.json")
    rel_id2label = {v:k for k,v in rel_label2id.items()}
    n_pred = 0
    n_gold = 0
    n_correct = 0
    count = 0
    total = 0
    count_rel = 0
    total_rel = 0
    for i in tqdm(range(len(test_data))):
        dic = {"doc_key": test_data[i]["doc_key"], "tokens": test_data[i]["tokens"],"entities": test_data[i]["entities"],"relations": test_data[i]["relations"]}
        ent_start2end = {i["start"]:i["end"] for i in dic["entities"]}
        dic["ground_truth"] = []
        for j in dic["relations"]:
            if j["label"] == "no_relation":
                continue
            dic["ground_truth"].append({"subj": " ".join(dic["tokens"][j["subj"]:ent_start2end[j["subj"]]+1]), "obj": " ".join(dic["tokens"][j["obj"]:ent_start2end[j["obj"]]+1]), "label": j["label"]})
        dic["predicted"] = []
        dic["error_list"] = []
        data_list = test_data[i]["train_data"]
        for i in range(len(data_list)):
            model.eval()
            with torch.no_grad():
                data_dic = data_list[i]
                out = tokenizer(" ".join(data_dic["typed_tokens"]),
                                max_length=args.max_length,
                                padding="max_length",
                                truncation=True,
                                add_special_tokens=True,
                                return_tensors="pt")
                t2i = tokenizer.convert_ids_to_tokens(out["input_ids"].squeeze().tolist())
                sub_idx = [t2i.index(i) for i in t2i if "<SUBJ_START_" in i]
                assert len(sub_idx) == 1
                obj_idx = [t2i.index(i) for i in t2i if "<OBJ_START_" in i]
                assert len(obj_idx) == 1
                sub_idx, obj_idx = sub_idx[0], obj_idx[0]
                input_ids = out["input_ids"]
                attention_mask = out["attention_mask"]
                token_type_ids = out["token_type_ids"]
                input_ids, attention_mask, token_type_ids = input_ids.to(device), attention_mask.to(device), token_type_ids.to(device)
                sub_idx, obj_idx = torch.tensor([sub_idx]).to(device), torch.tensor([obj_idx]).to(device)
                logits = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, sub_idx=sub_idx, obj_idx=obj_idx)
                pred = int(logits.argmax(dim=1))
                if rel_id2label[pred] != "no_relation":
                    n_pred += 1
                    dic["predicted"].append({"subj": " ".join(data_dic["typed_tokens"][data_dic["s_start"]+1:data_dic["s_end"]]), "obj": " ".join(data_dic["typed_tokens"][data_dic["o_start"]+1:data_dic["o_end"]]), "label": rel_id2label[pred]})
                label = rel_label2id[data_dic["rel"]]
                if rel_id2label[label] != "no_relation":
                    n_gold += 1
                    if label == pred:count_rel += 1
                    total_rel += 1
                if label == pred:
                    count += 1
                else:
                    dic["error_list"].append({"subj": " ".join(data_dic["typed_tokens"][data_dic["s_start"]+1:data_dic["s_end"]]), "obj": " ".join(data_dic["typed_tokens"][data_dic["o_start"]+1:data_dic["o_end"]]), "label": data_dic["rel"], "predict": rel_id2label[pred]})
                if (rel_id2label[pred] != "no_relation") and (rel_id2label[label] != "no_relation") and (pred == label):
                    n_correct += 1
                total += 1
        dic.pop("entities")
        dic.pop("relations")
        if not os.path.exists(f"./results/{args.dataset}"):os.makedirs(f"./results/{args.dataset}",exist_ok=True)
        with open(f"./results/{args.dataset}/{args.result_file_name}_fold{fold_round}.json","a") as f:
                f.write(str(dic)+"\n")
    prec = n_correct/n_pred
    recall = n_correct/n_gold
    f1 = 2*prec*recall / (prec + recall)
    log_file.info(f"*****fold_round:{fold_round}, test_Accuracy_precise: {count/total}, test_Accuracy_rel: {count_rel/total_rel}, precision: {prec}, recall : {recall}, f1: {f1}*****")

