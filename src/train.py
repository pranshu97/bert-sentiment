import warnings
warnings.filterwarnings("ignore")

import config
import dataset
from model import BertModel
import engine

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from transformers import AdamW, get_linear_schedule_with_warmup

def run():
    df = pd.read_csv(config.DATASET)
    df.sentiment = df.sentiment.apply(lambda x: 0 if x=='negative' else 1 if x=='neutral' else 2)

    negative_idx = np.where(df.sentiment==1)[0]
    np.random.shuffle(negative_idx)
    drop_idx = negative_idx[:len(negative_idx)//4]
    df = df.drop(drop_idx,axis=0)

    df_train, df_val = train_test_split(df,test_size=0.2,stratify=df.sentiment.values,shuffle=True,random_state=7)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    train_data = dataset.TweetDataset(df_train.text.values,df_train.sentiment.values)
    val_data = dataset.TweetDataset(df_val.text.values,df_val.sentiment.values)

    train_loader = torch.utils.data.DataLoader(train_data,batch_size=config.TRAIN_BATCH_SIZE,num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data,batch_size=config.VAL_BATCH_SIZE,num_workers=4)

    device = torch.device('cuda')

    model = BertModel().to(device)

    # for param in model.bert.parameters():
    #     param.requires_grad = False

    opt_param = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_parameters = [
        {
            'params':[p for n,p in opt_param if not any(nd in n for nd in no_decay)],
            'weight_decay':0.001
        },
        {
            'params':[p for n,p in opt_param if any(nd in n for nd in no_decay)],
            'weight_decay':0.0
        }
    ]

    num_training_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE*config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,num_warmup_steps=100,num_training_steps=num_training_steps)

    best_acc = 0

    for epoch in range(config.EPOCHS):
        train_loss = engine.train(train_loader, model, optimizer, scheduler, device)
        outputs,targets,loss = engine.eval(val_loader, model, device)
        outputs = outputs.argmax(axis=1)
        acc = accuracy_score(outputs, targets)
        f1 = f1_score(outputs,targets,average='weighted')
        print(f'Epoch:{epoch+1}, Train Loss: {train_loss}, Validation Loss: {loss}\nAccuracy: {acc}, F1 Score: {f1}\n')
        if acc>best_acc:
            torch.save(model, config.MODEL_PATH)
            best_acc = acc
            print('Accuracy improved, model saved.\n')

if __name__=='__main__':
	run()