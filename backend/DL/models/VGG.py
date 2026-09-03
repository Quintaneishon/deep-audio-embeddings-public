import torch.nn as nn
import torchaudio
import warnings

from backend.DL.modules import Res_2d, Res_2d_simple

class VGG_Res(nn.Module):
    def __init__(self,
                 n_channels=128,
                 sample_rate=16000,
                 n_fft=512,
                 f_min=0.0,
                 f_max=8000.0,
                 n_mels=128,
                 n_class=50,
                 use_simple_res=False):
        super(VGG_Res, self).__init__()

        # Choose the appropriate residual block based on architecture type
        ResBlock = Res_2d_simple if use_simple_res else Res_2d

        # Suppress mel filterbank warning (n_mels=128 is high for n_fft=512 but intentional)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, message='.*mel filterbank.*')
            self.spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                            n_fft=n_fft,
                                                            f_min=f_min,
                                                            f_max=f_max,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        self.spec_bn = nn.BatchNorm2d(1)

        self.layer1 = ResBlock(1, n_channels, stride=2)
        self.layer2 = ResBlock(n_channels, n_channels, stride=2)
        self.layer3 = ResBlock(n_channels, n_channels*2, stride=2)
        self.layer4 = ResBlock(n_channels*2, n_channels*2, stride=2)
        self.layer5 = ResBlock(n_channels*2, n_channels*2, stride=2)
        self.layer6 = ResBlock(n_channels*2, n_channels*2, stride=2)
        self.layer7 = ResBlock(n_channels*2, n_channels*4, stride=2)

        self.dense1 = nn.Linear(n_channels*4, n_channels*4)
        self.bn = nn.BatchNorm1d(n_channels*4)
        self.dense2 = nn.Linear(n_channels*4, n_class)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch_size, num_samples]

        x = self.spec(x)
        x = self.to_db(x)
        x = x.unsqueeze(1)
        x = self.spec_bn(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = x.squeeze(2)

        if x.size(-1) != 1:
            x = nn.MaxPool1d(x.size(-1))(x)
        x = x.squeeze(2)
        # x: [batch_size, n_channels*4]

        x = self.dense1(x)
        x = self.bn(x)

        embedding = self.relu(x)

        return embedding
