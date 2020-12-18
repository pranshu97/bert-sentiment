import config
import transformers
import torch.nn as nn

class BertModel(nn.Module):

	def __init__(self):
		super(BertModel,self).__init__()

		self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
		self.dropout = nn.Dropout(0.5)
		self.linear = nn.Linear(768,3)

	def forward(self,ids,mask,token_type_ids):
		_, out = self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids,return_dict=False)
		out = self.dropout(out)
		out = self.linear(out)
		return out
