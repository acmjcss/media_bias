import torch
import sys
import os
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score
from datasets import Dataset, load_metric
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
import random
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score
#2758 635 0.888

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf`` (if
    installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available

set_seed(1)

torch.cuda.empty_cache()

tokenizer = AutoTokenizer.from_pretrained("gchhablani/bert-base-cased-finetuned-mnli")

model = AutoModelForSequenceClassification.from_pretrained("gchhablani/bert-base-cased-finetuned-mnli")

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

#data_folder = sys.argv[1]#
data_folder = '/home/name/media_bias/allinone5/'
#output_folder = sys.argv[2]#
output_folder = '/home/name/media_bias/temp5/'
# topic = sys.argv[1]
# data_folder = f'/home/name/media_bias/all3one/dataset/{topic}'
# output_folder = f'/home/name/media_bias/all3one/dataset/{topic}'
aspects_file = os.path.join(data_folder,"relations_aspects.txt")
sents_file = os.path.join(data_folder,"relations_sentiments.txt")
idx_aspects_dict = dict()
aspects_idx_dict = dict()
idx_sents_dict = dict()
sents_idx_dict = dict()

topics = ['cab_bill','demonetization','farm_law','covid']
topic_aspects = {'cab_bill':set(),'demonetization':set(),'farm_law':set(),'covid':set()}
for topic in topics:
    with open(os.path.join(data_folder,f'relations_aspects_{topic}.txt')) as f:
        lines = [line.rstrip() for line in f]
        lines = [line for line in lines if line]
        i = 0
        for aspect in lines:
            topic_aspects[topic].add(aspect)

with open(aspects_file,"r") as f:
    lines = [line.rstrip() for line in f]
    lines = [line for line in lines if line]
    i = 0
    for aspect in lines:
        idx_aspects_dict[i] = aspect
        aspects_idx_dict[aspect] = i
        i+=1
    print("aspects_dict:")
    print(idx_aspects_dict)
    print(len(list(aspects_idx_dict.keys())))

with open(sents_file,"r") as f:
    lines = [line.rstrip() for line in f]
    lines = [line for line in lines if line]
    i = 0
    for sent in lines:
        idx_sents_dict[i] = sent.lower()
        sents_idx_dict[sent.lower()] = i
        i+=1
    print("sents_dict:")
    print(idx_sents_dict)
#2649 612 0.88988
#1083 256 0.77
def find_topic(i, train):
    if train:
        if i<=287:
            return 'cab_bill'
        elif i<=496:
            return 'demonetization'
        elif i<=891:
            return 'farm_law'
        else:
            return 'covid'
    else:
        if i<=99:
            return 'farm_law'
        elif i<=164:
            return 'cab_bill'
        elif i<=208:
            return 'demonetization'
        else:
            return 'covid'

def prepare_data(src_file, trg_file, aspects_idx, sents_idx):
    src_lines = open(src_file, encoding="utf-8").readlines()#
    trg_lines = open(trg_file, encoding="utf-8").readlines()#
    texts = []
    labels = []
    sents = []
    for i in range(len(src_lines)):
        src_line = src_lines[i].strip()
        trg_line = trg_lines[i].strip()

        current_labels = [0 for _ in aspects_idx]#all the unique aspects
        current_sents  = [0 for _ in aspects_idx]#aspect specific sentiments

        parts = trg_line.split("|")
        for part in parts:
            elements = part.split(";")
            aspect = elements[2]
            idx = aspects_idx[aspect]
            try:
                current_labels[idx] = 1
            except:
                print(idx)

            sent = elements[5].lower()
            sent_idx = sents_idx[sent]
            current_sents[idx] = sent_idx

        texts.append(src_line)
        labels.append(current_labels)
        sents.append(current_sents)
    return texts, labels, sents

def prepare_model_data(texts, labels, hypothesis_placeholder, idx_aspects):
    sequences = []
    hypotheses= []
    outputs   = []
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        for j in range(len(label)):
            sequences.append(text)
            hypotheses.append(hypothesis_placeholder.format(idx_aspects[j]))
            if label[j]==1:
                outputs.append(0)
            else:
                outputs.append(2)
    return sequences, hypotheses, outputs


def prepare_model_data_v2(texts, labels, hypothesis_placeholder, idx_aspects):
    sequences = []
    hypotheses= []
    outputs   = []
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        for j in range(len(label)):
            if label[j]==1:
                sequences.append(text)
                hypotheses.append(hypothesis_placeholder.format(idx_aspects[j]))
                outputs.append(0)
            elif random.randint(0,19)==0:
                sequences.append(text)
                hypotheses.append(hypothesis_placeholder.format(idx_aspects[j]))
                outputs.append(2)

    return sequences, hypotheses, outputs

def prepare_model_data_v3(texts, labels, hypothesis_placeholder, idx_aspects, train):
    sequences = []
    hypotheses= []
    outputs   = []
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        for j in range(len(label)):
            if label[j]==1:
                sequences.append(text)
                hypotheses.append(hypothesis_placeholder.format(idx_aspects[j]))
                outputs.append(0)
            else:
                topic = find_topic(i, train)
                if random.randint(0,1) and idx_aspects[j] in topic_aspects[topic]:
                    sequences.append(text)
                    hypotheses.append(hypothesis_placeholder.format(idx_aspects[j]))
                    outputs.append(2)

    return sequences, hypotheses, outputs

def prepare_model_data_sent(texts, labels, sents, hypothesis_placeholder, idx_aspects, idx_sents):
    sequences = []
    hypothesis= []
    outputs   = []
    for i in range(len(texts)):
        text = texts[i]
        label = labels[i]
        sent = sents[i]
        for j in range(len(label)):
            if label[j]:
                sequences.append(text)
                hypothesis.append(hypothesis_placeholder.format(idx_aspects[j]))
                outputs.append(sent[j])
                # sequences.append(text)
                # hypothesis.append(hypothesis_placeholder.format(idx_aspects[j],idx_sents[0]))
                # sequences.append(text)
                # hypothesis.append(hypothesis_placeholder.format(idx_aspects[j],idx_sents[2]))
                # outputs.append(sent[j])
    return sequences, hypothesis, outputs




#Creating the dataset for the neural network
src_train_file = os.path.join(data_folder,"train_bert.sent")
trg_train_file = os.path.join(data_folder,"train_bert.pointer")
src_val_file = os.path.join(data_folder,"dev_bert.sent")
trg_val_file = os.path.join(data_folder,"dev_bert.pointer")
#src_test_file = os.path.join(data_folder,"test_bert.sent")
#trg_test_file = os.path.join(data_folder,"test_bert.pointer")
train_data_texts, train_data_labels, train_data_sents = prepare_data(src_train_file, trg_train_file, aspects_idx_dict, sents_idx_dict)
val_data_texts, val_data_labels, val_data_sents = prepare_data(src_val_file, trg_val_file, aspects_idx_dict, sents_idx_dict)
#test_data_texts, test_data_labels, test_data_sents = prepare_data(src_test_file, trg_test_file,aspects_idx_dict, sents_idx_dict)

columns = ["sentence","hypothesis","label"]

train_sequences, train_hypotheses, train_true_outputs = prepare_model_data_v3(train_data_texts, train_data_labels, "The sentence is about {}", idx_aspects_dict, 1)

print(len(train_sequences))

#preprocess for randomization
suffling_list = np.arange(len(train_sequences))
np.random.shuffle(suffling_list)
suffling_list = list(suffling_list)
print(len(suffling_list))


#randomization
train_sequences = [train_sequences[ri] for ri in suffling_list]
train_hypotheses = [train_hypotheses[ri] for ri in suffling_list]
train_true_outputs = [train_true_outputs[ri] for ri in suffling_list]

print(len(train_sequences))
print(len(train_hypotheses))
print(len(train_true_outputs))

'''
#data count variation study
train_sequences = train_sequences[:1500]
train_hypotheses = train_hypotheses[:1500]
train_true_outputs = train_true_outputs[:1500]
'''

labels = train_true_outputs
train_df = pd.DataFrame(columns=columns)
train_df["sentence"] = train_sequences
train_df["hypothesis"] = train_hypotheses
train_df["label"] = labels
# train_df.to_csv(f'/home/name/media_bias/all3one/dataset/{topic}/train.csv')
train_df.to_csv(data_folder+'all_aspect_train.csv')
train_dataset = Dataset.from_pandas(train_df)

print("\ntraining dataset size={}".format(len(train_sequences)))


val_sequences, val_hypotheses, val_true_outputs = prepare_model_data_v3(val_data_texts, val_data_labels, "The sentence is about {}", idx_aspects_dict, 0)
labels = val_true_outputs
val_df = pd.DataFrame(columns=columns)
val_df["sentence"] = val_sequences
val_df["hypothesis"] = val_hypotheses
val_df["label"] = labels
# val_df.to_csv(f'/home/name/media_bias/all3one/dataset/{topic}/dev.csv')
val_df.to_csv(data_folder+'all_aspect_dev.csv')
val_dataset = Dataset.from_pandas(val_df)

print("\nvalidation dataset size={}".format(len(val_sequences)))

def tokenize_function(example):
    return tokenizer(example["sentence"], example["hypothesis"], truncation=True)

train_datasets = train_dataset.map(tokenize_function, batched=True)
val_datasets = val_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


train_datasets = train_datasets.remove_columns(["sentence","hypothesis"])
val_datasets = val_datasets.remove_columns(["sentence","hypothesis"])

train_datasets = train_datasets.rename_column("label","labels")
val_datasets = val_datasets.rename_column("label","labels")

train_datasets.set_format("torch")
val_datasets.set_format("torch")

train_dataloader = DataLoader(train_datasets,shuffle=True,batch_size=8, collate_fn=data_collator)
val_dataloader = DataLoader(val_datasets,batch_size=8, collate_fn=data_collator)

optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 20
num_training_steps = num_epochs * len(train_dataloader)#
lr_scheduler = get_scheduler(name="linear",optimizer=optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)

print("\ntraining steps={}".format(num_training_steps))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
#metric = load_metric("accuracy")###
progress_bar = tqdm(range(num_training_steps))
#cal_loss = nn.NLLLoss()

best_dev_score = -1.0
best_epoch = -1
for epoch in range(num_epochs):
    print('Epoch: {}'.format(epoch))
    train_loss_val = 0.0
    model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k,v in batch.items()}
        outputs = model(**batch)
        #actual_labels = batch['labels']#actual labels
        #pred_logits = outputs.logits#predicted output scores
        #loss = cal_loss(pred_logits, actual_labels)
        loss = outputs.loss#loss calculate
        loss.backward()#backpropagate

        optimizer.step()#parameter update
        lr_scheduler.step()#lr update
        optimizer.zero_grad()#optimizer initialization
        progress_bar.update(1)
        train_loss_val+=loss.item()
    print('Train loss: {}'.format(train_loss_val))

    model.eval()
    f_score = 0.0
    total_labels = []
    total_preds = []
    for batch in val_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        preds_post = torch.where(preds>0,2,0)
        total_labels.extend(batch['labels'].cpu())
        total_preds.extend(preds_post.cpu())
        #metric.add_batch(predictions=preds_post,references=batch["labels"])###
    f_score = f1_score(total_labels, total_preds, average='macro')
    #score = metric.compute()###
    #print('Dev accuracy: {}'.format(score))###
    print('Dev accuracy: {}'.format(f_score))
    #dev_score = score['accuracy']###
    dev_score = f_score
    if dev_score > best_dev_score:
        best_dev_score = dev_score
        best_epoch = epoch
        model.save_pretrained(os.path.join(output_folder,"aspect_journal"))
        best_dev_precision = precision_score(total_labels, total_preds, average='macro')
        best_dev_recall = recall_score(total_labels, total_preds, average='macro')
        print('Aspect identification model saved............................')
    if epoch + 1 - best_epoch >=5 :
        break
print('Best Epoch= {}'.format(best_epoch))
print('Accuracy= {}'.format(best_dev_score))
print('Precision= {}'.format(best_dev_precision))
print('Recall= {}'.format(best_dev_recall))
