import torch
import torch.nn as nn
import torchaudio

from backend.DL.modules import Conv_1d, Conv_V, Conv_H

class Musicnn(nn.Module):
    def __init__(self,
                sample_rate=16000,
                n_fft=512,
                f_min=0.0,
                f_max=8000.0,
                n_mels=96,
                n_class=50,
                dataset='mtat'):
        super(Musicnn, self).__init__()

        # Spectrogram
        self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                         n_fft=n_fft,
                                                         f_min=f_min,
                                                         f_max=f_max,
                                                         n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        # Pons front-end
        m1 = Conv_V(1, 204, (int(0.7*96), 7))
        m2 = Conv_V(1, 204, (int(0.4*96), 7))
        m3 = Conv_H(1, 51, 129)
        m4 = Conv_H(1, 51, 65)
        m5 = Conv_H(1, 51, 33)
        self.layers = nn.ModuleList([m1, m2, m3, m4, m5])

        # Pons back-end
        backend_channel= 512 if dataset=='msd' else 64
        self.layer1 = Conv_1d(561, backend_channel, 7, 1, 1)
        self.layer2 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)
        self.layer3 = Conv_1d(backend_channel, backend_channel, 7, 1, 1)

        # Dense
        dense_channel = 500 if dataset=='msd' else 200
        self.dense1 = nn.Linear((561+(backend_channel*3))*2, dense_channel)
        self.bn = nn.BatchNorm1d(dense_channel)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(dense_channel, n_class)

    def forward(self, x):
        # x: [batch_size, num_samples]

        # Spectrogram
        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)
        # x: [batch_size, 1, n_mels, time_frames]

        # Pons front-end: Filtros horizontales y verticales
        out = torch.cat([layer(x) for layer in self.layers], dim=1)
        # out: [batch_size, 561, 1, time_frames]

        # Pons back-end: Convoluciones 1D con Res-connections
        out = out.squeeze(2) # [batch_size, 561, time_frames]
        length = out.size(2)

        # Res-connections: res2 = layer2(res1) + res1
        res1 = self.layer1(out)
        res2 = self.layer2(res1) + res1
        res3 = self.layer3(res2) + res2
        out = torch.cat([out, res1, res2, res3], 1)

        # Global pooling: MaxPool y AvgPool sobre la dimensión del tiempo
        mp = nn.MaxPool1d(length)(out)
        avgp = nn.AvgPool1d(length)(out)
        out = torch.cat([mp, avgp], dim=1).squeeze(2)

        # Capas Densas
        # Primera capa densa (Penúltima Capa)
        out = self.relu(self.bn(self.dense1(out)))

        # *** EMBEDDING: CAPTURA DE LA PENÚLTIMA CAPA ***
        embedding = out

        return embedding
