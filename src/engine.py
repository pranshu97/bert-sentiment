import config
from tqdm import tqdm
import torch.nn as nn
import torch
import numpy as np

def loss_fn(outputs,targets):
	return nn.CrossEntropyLoss()(outputs.argmax(dim=1),targets)

def train(data_loader, model, optimizer, scheduler,device):
	model.train()
	for i,data in tqdm(enumerate(data_loader),total=len(data_loader)):
		ids = data['ids'].to(device)
		token_type_ids = data['token_type_ids'].to(device)
		mask = data['mask'].to(device)
		targets = data['targets'].to(device)

		optimizer.zero_grad()
		out = model(ids,mask,token_type_ids)
		
		loss = loss_fn(out,targets)
		loss.backward()

		optimizer.step()
		scheduler.step()


def eval(data_loader, model, device):
	model.eval()

	total_targets = []
	total_outputs = []
	total_loss = []
	with torch.no_grad():
		for i,data in tqdm(enumerate(data_loader),total=len(data_loader)):
			ids = data['ids'].to(device)
			token_type_ids = data['token_type_ids'].to(device)
			mask = data['mask'].to(device)
			targets = data['targets'].to(device)

			out = model(ids,mask,token_type_ids)
			out = torch.softmax(out)
			
			loss = loss_fn(out,targets)

			total_targets.extend(targets.cpu().detach().numpy().tolist())
			total_outputs.extend(out.cpu().detach().numpy().tolist())
			total_loss.append(loss)

		total_loss = np.array(total_loss)
		avg_loss = sum(total_loss)/len(total_loss)

	return total_outputs,total_targets,avg_loss

