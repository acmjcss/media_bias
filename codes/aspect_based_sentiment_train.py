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

#1137 266 0.774

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

# del model
# del pytorch_model
# del trainer
torch.cuda.empty_cache()

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

data_folder = '/home/name/media_bias/allinone5/'
# topic = sys.argv[1]
# data_folder = f'/home/name/media_bias/all3one/dataset/{topic}'
output_folder = '/home/name/media_bias/temp5/'
# output_folder = f'/home/name/media_bias/all3one/dataset/{topic}'
aspects_file = os.path.join(data_folder,"relations_aspects.txt")
sents_file = os.path.join(data_folder,"relations_sentiments.txt")
idx_aspects_dict = dict()
aspects_idx_dict = dict()
idx_sents_dict = dict()
sents_idx_dict = dict()

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


def prepare_data(src_file, trg_file, aspects_idx, sents_idx):
    src_lines = open(src_file, encoding="utf-8").readlines()
    trg_lines = open(trg_file, encoding="utf-8").readlines()
    texts = []
    labels = []
    sents = []
    for i in range(len(src_lines)):
        src_line = src_lines[i].strip()
        trg_line = trg_lines[i].strip()

        current_labels = [0 for _ in aspects_idx]
        current_sents  = [0 for _ in aspects_idx]

        parts = trg_line.split("|")
        for part in parts:
            elements = part.split(";")
            aspect = elements[2]#aspect phrase
            idx = aspects_idx[aspect]#aspect index
            current_labels[idx] = 1#set aspect index

            sent = elements[5].lower()#sentiment phrase
            sent_idx = sents_idx[sent]#sentiment index
            current_sents[idx] = sent_idx# update sentiment index

        texts.append(src_line)
        labels.append(current_labels)
        sents.append(current_sents)
    return texts, labels, sents


def prepare_model_data_sent(texts, labels, sents, hypothesis_placeholder, idx_aspects, idx_sents):
    sequences = []
    hypothesis= []
    outputs   = []
    for i in range(len(texts)):#for each sentence
        text = texts[i]#sentence
        label = labels[i]#list of aspect specific presence[0,1,1,0,0,0]
        sent = sents[i]#list if aspect specific sentiment [0/1/2,0/1/2,0/1/2,...]
        for j in range(len(label)):#for each aspect
            if label[j]:#if it is present
                sequences.append(text)#premise
                hypothesis.append(hypothesis_placeholder.format(idx_aspects[j]))#hypothesis
                outputs.append(sent[j])#0/1/2
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

train_sent_sequences, train_sent_hypotheses, train_sent_true_outputs = prepare_model_data_sent(train_data_texts, train_data_labels, train_data_sents, "{} has positive impact", idx_aspects_dict, idx_sents_dict)
#train_sent_sequences, train_sent_hypotheses, train_sent_true_outputs = prepare_model_data_sent(train_data_texts, train_data_labels, train_data_sents, "{} is depicted {}ly", idx_aspects_dict, idx_sents_dict)

#preprocess for randomization
suffling_list = np.arange(len(train_sent_sequences))
np.random.shuffle(suffling_list)
suffling_list = list(suffling_list)
print(len(suffling_list))


#randomization
train_sent_sequences = [train_sent_sequences[ri] for ri in suffling_list]
train_sent_hypotheses = [train_sent_hypotheses[ri] for ri in suffling_list]
train_sent_true_outputs = [train_sent_true_outputs[ri] for ri in suffling_list]


#data count variation study
train_sent_sequences = train_sent_sequences[:800]
train_sent_hypotheses = train_sent_hypotheses[:800]
train_sent_true_outputs = train_sent_true_outputs[:800]




labels = train_sent_true_outputs
train_sent_df = pd.DataFrame(columns=columns)
train_sent_df["sentence"] = train_sent_sequences
train_sent_df["hypothesis"] = train_sent_hypotheses
train_sent_df["label"] = labels
train_sent_df.to_csv(data_folder+'all_sentiment_train.csv')
train_sent_dataset = Dataset.from_pandas(train_sent_df)

print("\ntraining sentiment dataset size={}".format(len(train_sent_sequences)))


val_sent_sequences, val_sent_hypotheses, val_sent_true_outputs = prepare_model_data_sent(val_data_texts, val_data_labels, val_data_sents, "{} has positive impact", idx_aspects_dict, idx_sents_dict)
#val_sent_sequences, val_sent_hypotheses, val_sent_true_outputs = prepare_model_data_sent(val_data_texts, val_data_labels, val_data_sents, "{} is depicted {}ly", idx_aspects_dict, idx_sents_dict)
labels = val_sent_true_outputs
val_sent_df = pd.DataFrame(columns=columns)
val_sent_df["sentence"] = val_sent_sequences
val_sent_df["hypothesis"] = val_sent_hypotheses
val_sent_df["label"] = labels
val_sent_df.to_csv(data_folder+'all_sentiment_dev.csv')
val_sent_dataset = Dataset.from_pandas(val_sent_df)
print("\nvalidation sentiment dataset size={}".format(len(val_sent_sequences)))


tokenizer = AutoTokenizer.from_pretrained("gchhablani/bert-base-cased-finetuned-mnli")
model_sent = AutoModelForSequenceClassification.from_pretrained("gchhablani/bert-base-cased-finetuned-mnli")

def tokenize_function(example):
    return tokenizer(example["sentence"], example["hypothesis"], truncation=True)


train_sent_datasets = train_sent_dataset.map(tokenize_function, batched=True)
val_sent_datasets = val_sent_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_sent_datasets = train_sent_datasets.remove_columns(["sentence","hypothesis"])
val_sent_datasets = val_sent_datasets.remove_columns(["sentence","hypothesis"])

train_sent_datasets = train_sent_datasets.rename_column("label","labels")
val_sent_datasets = val_sent_datasets.rename_column("label","labels")

train_sent_datasets.set_format("torch")
val_sent_datasets.set_format("torch")

train_sent_dataloader = DataLoader(train_sent_datasets, shuffle=True, batch_size=8, collate_fn=data_collator)
val_sent_dataloader = DataLoader(val_sent_datasets, batch_size=8, collate_fn=data_collator)

num_epochs = 20#epoch
num_training_steps = num_epochs * len(train_sent_dataloader)#steps
print("\ntraining steps={}".format(num_training_steps))
optimizer = AdamW(model_sent.parameters(), lr=5e-5)#optimizer
lr_scheduler = get_scheduler(name="linear",optimizer=optimizer,num_warmup_steps=0,num_training_steps=num_training_steps)#schedular

model_sent.to(device)
metric = load_metric("f1")


progress_bar = tqdm(range(num_training_steps))
best_dev_score = -1.0
best_epoch = -1

for epoch in range(num_epochs):
    print('Epoch: {}'.format(epoch))
    train_loss_val = 0.0
    model_sent.train()
    for batch in train_sent_dataloader:
        batch = {k: v.to(device) for k,v in batch.items()}
        outputs = model_sent(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        train_loss_val+=loss.item()
    print('Train loss: {}'.format(train_loss_val))

    model_sent.eval()
    for batch in val_sent_dataloader:
        batch = {k:v.to(device) for k,v in batch.items()}
        with torch.no_grad():
            outputs = model_sent(**batch)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=preds,references=batch["labels"])
    score = metric.compute(average='macro')
    print('Dev accuracy: {}'.format(score))
    dev_score = score['f1']
    if dev_score > best_dev_score:
        best_dev_score = dev_score
        best_epoch = epoch
        model_sent.save_pretrained(os.path.join(output_folder,"sentiment_900_journal"))
        print('Sentiment classification model saved............................')
    if epoch + 1 - best_epoch >=5 :
        break
print('Best Epoch= {}'.format(best_epoch))
print('F-score= {}'.format(best_dev_score))
