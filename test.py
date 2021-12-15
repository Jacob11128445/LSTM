from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
 
 
'''
下面的split_sequence（）函数实现了这种行为，并将给定的单变量序列分成多个样本，其中每个样本具有指定的时间步长，输出是单个时间步。
'''
# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)
 
 
if __name__ == '__main__':
 
    # define input sequence
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    print (raw_seq)
    # choose a number of time steps
    n_steps = 3
    # split into samples
    X, y = split_sequence(raw_seq, n_steps)
    print (X, y)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))  # 隐藏层，输入，特征维
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X, y, epochs=300, batch_size=1, verbose=2)  # 迭代次数，批次数，verbose决定是否显示每次迭代
    # demonstrate prediction
    x_input = array([70, 80, 90])
    x_input = x_input.reshape((1, n_steps, n_features))
    yhat = model.predict(x_input, verbose=0)
    print (x_input, yhat)
    print(yhat)

#################################################################################################
from unittest import TestCase
import torch as t
import torch.nn as nn
from torch.optim import Adam
from pandas_ml_quant.pytorch.custom_loss import SoftDTW
from pandas_ml_quant_test.config import DF_TEST
from pandas_ml_quant import PostProcessedFeaturesAndLabels
from pandas_ml_utils import AutoEncoderModel, FeaturesAndLabels
from pandas_ml_utils.ml.model.pytoch_model import PytorchModel
class TestCustomLoss(TestCase):
    def test_soft_dtw_loss(self):
        df = DF_TEST[["Close"]][-21:].copy()
        class LstmAutoEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_size = 1
                self.seq_size=10
                self.hidden_size = 2
                self.num_layers = 1
                self.num_directions = 1
                self._encoder =\
                    nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                           batch_first=True)
                self._decoder =\
                    nn.RNN(input_size=self.hidden_size, hidden_size=self.input_size, num_layers=self.num_layers,
                           batch_first=True)
            def forward(self, x):
                # make sure to treat single elements as batches
                x = x.view(-1, self.seq_size, self.input_size)
                batch_size = len(x)
                hidden_encoder = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
                hidden_decoder = nn.Parameter(t.zeros(self.num_layers * self.num_directions, batch_size, self.input_size))

                x, _ = self._encoder(x, hidden_encoder)
                # print(x.shape)
                x = t.repeat_interleave(x[:,-2:-1], x.shape[1], dim=1)
                x, hidden = self._decoder(x, hidden_decoder)
                return x

            def encoder(self, x):
                x = x.reshape(-1, self.seq_size, self.input_size)
                batch_size = len(x)
                with t.no_grad():
                    hidden = nn.Parameter(
                        t.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size))
                    return self._encoder(t.from_numpy(x).float(), hidden)[0].numpy()

                    # return last element of sequence
                    return self._encoder(t.from_numpy(x).float(), hidden)[0].numpy()[:,-1]

            def decoder(self, x):
                x = x.reshape(-1, self.seq_size, self.hidden_size)
                batch_size = len(x)
                with t.no_grad():
                    hidden = nn.Parameter(
                        t.zeros(self.num_layers * self.num_directions, batch_size, self.input_size))
                    return self._decoder(t.from_numpy(x).float(), hidden)[0].numpy()
        model = AutoEncoderModel(
            PytorchModel(
                PostProcessedFeaturesAndLabels(df.columns.to_list(), [lambda df: df.ta.rnn(10)],
                                               df.columns.to_list(), [lambda df: df.ta.rnn(10)]),
                LstmAutoEncoder,
                SoftDTW,
                Adam
            ),
            ["condensed"],
            ["condensed-a", "condensed-b"],
            lambda m: m.module.encoder,
            lambda m: m.module.decoder,
        )

        fit = df.model.fit(model, epochs=100)
        print(fit)
        print(fit.test_summary.df)

        encoded = df.model.predict(fit.model.as_encoder())
        print(encoded)