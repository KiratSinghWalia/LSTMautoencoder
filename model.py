import torch
torch.cuda.is_available()
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class Encoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=128):
    super(Encoder, self).__init__()
    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )
  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))
    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)
    return hidden_n.reshape((1, self.embedding_dim)),x

class Decoder(nn.Module):
  def __init__(self, seq_len, input_dim=128, n_features=6):
    super(Decoder, self).__init__()
    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features
    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )
    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    self.output_layer = nn.Linear(self.hidden_dim, n_features)
  def forward(self, x):

    x = x.unsqueeze(1).tile((1, self.seq_len, 1))
    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)


class LSTMAutoencoder(nn.Module):
  def __init__(self, seq_len, n_features, embedding_dim=128):
    super(LSTMAutoencoder, self).__init__()
    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)
  def forward(self, x):
    hidden_n, _ = self.encoder(x)
    x = self.decoder(hidden_n)
    return x


@torch.no_grad()
def estimate_loss():
  out={}
  model.eval()
  for split in ['train','val']:
    losses=torch.zeros(eval_iters)
    for i in range(eval_iters):
      x,y=get_baches(split)
      predictions=model(x)
      predictions=predictions.unsqueeze(0)
      loss=nn.MSELoss().to(device)
      loss=loss(predictions,y)
      losses[i]=loss.item()
    out[split]=losses.mean()
  model.train()
  return out

seq_len=_
n_features=_
model=LSTMAutoencoder(seq_len,n_features).to(device)
