from transformers import BertConfig, BertTokenizer, BertModel
bertname = "distilbert"
config = BertConfig.from_pretrained(bertname)
tokenizer = BertTokenizer.from_pretrained(bertname)
model = BertModel.from_pretrained(bertname)

