#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
import torch
import time
import json

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# In[ ]:


df = pd.read_excel('../dataset/ije_sa/ije_sa.xlsx')
train_data = pd.read_excel('../dataset/ije_sa/train_set.xlsx')
val_data = pd.read_excel('../dataset/ije_sa/validation_set.xlsx')
test_data = pd.read_excel('../dataset/ije_sa/test_set.xlsx')

# Length train, val, and test
print("Train: ",len(train_data))
print("Val: ",len(val_data))
print("Test: ",len(test_data))

tags = np.unique(df['label'])
num_labels = len(tags)
max_length = 128
label2id = {t: i for i, t in enumerate(tags)}
id2label = {i: t for i, t in enumerate(tags)}


# In[ ]:


def model_init(model_name):
    global tokenizer
    global data_collator
    global tr_model

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=128)
    
    return model, tokenizer


# In[ ]:


def tokenize_function(examples):
    # process the input sequence
    tokenized_input = tokenizer(examples["tweet"], 
                                truncation=True, 
                                padding='max_length', 
                                max_length=max_length)
    # process the labels
    tokenized_input['label'] = [label2id[lb] for lb in examples['label']]
    
    return tokenized_input


# In[ ]:


def preprocessing():
    X_train = Dataset.from_pandas(train_data)
    X_val = Dataset.from_pandas(val_data)
    X_test = Dataset.from_pandas(test_data)
    
    tokenized_train_data = X_train.map(tokenize_function, batched=True)
    tokenized_val_data = X_val.map(tokenize_function, batched=True)
    tokenized_test_data = X_test.map(tokenize_function, batched=True)
    
    return tokenized_train_data, tokenized_val_data, tokenized_test_data


# In[ ]:


def compute_metrics(p):
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    
    true_labels = [tags[l] for l in labels] 
    true_predictions = [tags[pr] for pr in pred]

    report = classification_report(true_labels, true_predictions, digits=4)
    acc = accuracy_score(y_true=true_labels, y_pred=true_predictions)
    rec = recall_score(y_true=true_labels, y_pred=true_predictions, average='macro')
    prec = precision_score(y_true=true_labels, y_pred=true_predictions, average='macro')
    f1 = f1_score(y_true=true_labels, y_pred=true_predictions, average='macro')

    print("Classification Report:\n{}".format(report))
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# In[ ]:


def train_model(model_name, output_dir, 
                learning_rate, train_batch_size, 
                eval_batch_size, num_epochs, weight_decay):
    model, tokenizer = model_init(model_name)
    train_tokenized, val_tokenized, test_tokenized = preprocessing()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 1,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        weight_decay=weight_decay,
        #push_to_hub=True,
        metric_for_best_model = "f1",
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(output_dir)
    #trainer.push_to_hub(commit_message="Training complete")


# In[ ]:


def predict_model(model_name):
    model, tokenizer = model_init(model_name)
    train_tokenized, val_tokenized, test_tokenized = preprocessing()

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    # trainer.predict(test_tokenized)
    # trainer.push_to_hub(commit_message="Test complete")
    print(trainer.predict(test_tokenized))


# In[ ]:


def main(model_name, output_dir, best_params):
    start = time.time()
     # load json file containing best params
    best_params = best_params

    with open(best_params, 'r') as js:
        data = json.load(js)

    print(data)

    # define best params
    learning_rate = data['learning_rate']
    train_batch_size = data['per_device_train_batch_size']
    eval_batch_size = data['per_device_eval_batch_size']
    weight_decay = data['weight_decay']

    # training
    train_model(model_name=model_name, 
                output_dir=output_dir,
                learning_rate=learning_rate,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                num_epochs=5,
                weight_decay=weight_decay)

    print('Training finished!')
    
    # prediction
    predict_model(model_name=f'{output_dir}')
    print('Prediction finished!')

    
    end = time.time()
    exec_time = (end - start) / 60
    print(f'Total time: {exec_time} minutes')

