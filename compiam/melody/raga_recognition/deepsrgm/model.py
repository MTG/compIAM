import torch.nn as nn
from compiam.melody.raga_recognition.deepsrgm.attention_layer import Attention


class deepsrgmModel(nn.Module):
    def __init__(
        self,
        rnn="lstm",
        input_length=5000,
        embedding_size=128,
        hidden_size=768,
        num_layers=1,
        num_classes=10,
        vocab_size=209,
        drop_prob=0.3,
    ):
        """DEEPSRGM model init class

        :param rnn: indicates whether to use an LSTM or a GRU
        :param input_length: length of input subsequence
        :param embedding_size: dim of the embedding for each element in input
        :param hidden_size: number of features in the hidden state of LSTM
        :param num_layers: number of LSTM layers
        :param num_classes: number of classes for classification task
        :param vocab_size: size of vocabulary for embedding layer
        :param drop_prob: dropour probability
        """

        super(deepsrgmModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        if rnn == "lstm":
            self.rnn = nn.LSTM(
                embedding_size,
                hidden_size,
                num_layers,
                dropout=drop_prob,
                batch_first=True,
            )
        elif rnn == "gru":
            self.rnn = nn.GRU(
                embedding_size,
                hidden_size,
                num_layers,
                dropout=drop_prob,
                batch_first=True,
            )

        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.attention_layer = Attention(hidden_size, input_length)

        self.fc1 = nn.Linear(hidden_size, 384)
        self.fc2 = nn.Linear(384, num_classes)

        # self.batchNorm1d = nn.BatchNorm1d(input_length)
        self.dropout = nn.Dropout(drop_prob)
        self.relu = nn.ReLU()

    def forward(self, x):
        # batch_size = x.size(0)
        embeds = self.embeddings(x)
        out, _ = self.rnn(embeds)
        # out = self.batchNorm1d(out)
        out = self.attention_layer(out)

        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)

        return out
