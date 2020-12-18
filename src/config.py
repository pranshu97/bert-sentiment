import transformers

MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
VAL_BATCH_SIZE = 16
EPOCHS = 10
BERT_PATH = "../input/bert_base_uncased/"
ACCUMULATION = 2
MODEL_PATH = "model.bin"
DATASET = "../input/train.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)