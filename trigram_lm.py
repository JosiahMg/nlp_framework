from dataset.trigram_dataset import TriGramDataset
from models.nnlm_lm import NNLMModel


dataset = TriGramDataset()
vocab_size = dataset.get_vocab_size()
seq_len = dataset.get_seq_len()
embed_size = 100
hidden_size = 20

trigram = NNLMModel(vocab_size, embed_size, seq_len, hidden_size)

dataset = TriGramDataset()
dataloader = dataset.dataloader()

trigram.fit(dataloader, dataloader)

inputs, targets = dataset.get_data()
preds = trigram.predict(inputs)

for data in inputs:
    print([dataset.idx2word[d.item()] for d in data])
print([dataset.idx2word[pred.cpu().item()] for pred in preds])
