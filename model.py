import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertModel

class NERModel(nn.Module):
    def __init__(self,tag_size=None, device='cpu', finetuning=False, bert_model = 'bert-base-cased'):
        super().__init__()
        self.ker_size = 2
        self.device = device
        self.bert = BertModel.from_pretrained(bert_model)
        self.conv1d = nn.Conv1d(768, 100, kernel_size=self.ker_size, padding=(self.ker_size-1, 0))
        self.mheadAtt = nn.MultiheadAttention(100, 1, dropout=0.8)
        self.finetuning = finetuning
        self.fc = nn.Linear(100, tag_size)

    def forward(self, x, y ):
        x = x.to(self.device)
        y = y.to(self.device)
        if self.finetuning:
            # print("->bert.train()")
            self.bert.train()
            encoded_layers, _ = self.bert(x)
            enc = encoded_layers[-1]
        else:
            self.bert.eval()
            with torch.no_grad():
                encoded_layers, _ = self.bert(x)
                enc = encoded_layers[-1]

        print(enc.shape)
        features_conv = self.conv1d(enc)
        print(features_conv.shape)

        features_att, _ = self.mheadAtt(features_conv)
        print(features_att.shape)


        logits = self.fc(features_att)
        y_hat = logits.argmax(-1)
        return logits, y, y_hat