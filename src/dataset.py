import config
import torch

class TweetDataset:
	
	def __init__(self,tweets,sentiments):
		self.tweets = tweets
		self.sentiments = sentiments
		self.tokenizer = config.TOKENIZER
		self.max_len = config.MAX_LEN

	def __len__(self):
		return len(self.tweets)

	def __getitem__(self,idx):
		tweet = str(self.tweets[idx])
		tweet = ' '.join(tweet.split())

		inp = self.tokenizer.encode_plus(tweet,None,add_special_tokens=True,max_length=self.max_len)

		ids = inp['input_ids']
		mask = inp['attention_mask']
		token_type_ids = inp['token_type_ids']

		padding_len = self.max_len-len(ids)

		ids = ids + ([0]*padding_len)
		mask = mask + ([0]*padding_len)
		token_type_ids = token_type_ids + ([0]*padding_len)

		return {
				'ids':torch.tensor(ids, dtype=torch.long),
				'mask':torch.tensor(mask, dtype=torch.long),
				'token_type_ids':torch.tensor(token_type_ids, dtype=torch.long),
				'targets':torch.tensor(self.sentiments[idx],dtype=torch.long)	
		}