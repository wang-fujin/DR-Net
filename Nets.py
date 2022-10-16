import torch
import torch.nn as nn
from DSBN import DomainSpecificBatchNorm1D

class Swish_act(nn.Module):
    def __init__(self):
        super(Swish_act, self).__init__()

    def forward(self, x):
        x = x * torch.sigmoid(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1)
        self.bn1 = DomainSpecificBatchNorm1D(output_channel)
        self.activation = Swish_act()
        self.conv2 = nn.Conv1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = DomainSpecificBatchNorm1D(output_channel)

        self.input_channel = input_channel
        self.output_channel = output_channel


        self.conv_skip = nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride)
        self.bn_skip = DomainSpecificBatchNorm1D(output_channel)

    def forward(self, x, domain_label):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out,domain_label)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out,domain_label)

        if self.input_channel != self.output_channel:
            residual = self.conv_skip(x)
            residual = self.bn_skip(residual,domain_label)
        out = out + residual
        out = self.activation(out)

        return out

class Encoder(nn.Module):
    '''
    (batch_size, Channel, Seq_len) ——> (batch_size, embedding_length)
    (batch_size, C, 128) --> (batch_size, 256)
    '''
    def __init__(self, input_channel=3,embedding_length=256):    # (batch_size, C, 128)
        super(Encoder, self).__init__()
        self.embedding_length = embedding_length

        self.conv1 = nn.Conv1d(input_channel, 16, kernel_size=1, stride=1, padding=0)  # batch_size, 16, 128
        self.bn1 = DomainSpecificBatchNorm1D(16)
        self.activation = Swish_act()



        self.layer1 = EncoderBlock(16, 32, stride=1)  # batch, 32, 128
        self.layer2 = EncoderBlock(32, 64, stride=2)  # batch, 64, 64
        self.layer3 = EncoderBlock(64, 128, stride=2)  # batch, 128, 32
        self.layer4 = EncoderBlock(128, 192, stride=2)  # batch, 192, 16



        self.linear = nn.Sequential(
            nn.Linear(192 * 16, 512),
            nn.Dropout(),
            nn.Linear(512, embedding_length)
        )

    def forward(self, x, domain_label):
        out = self.conv1(x)
        out = self.bn1(out,domain_label)
        out = self.activation(out)


        out = self.layer1(out,domain_label)
        out = self.layer2(out,domain_label)
        out = self.layer3(out,domain_label)
        out = self.layer4(out,domain_label)
        pred = self.linear(out.view(out.size(0), -1))
        return pred

    def output_dim(self):
        return self.embedding_length


class Discriminator(nn.Module):
    '''
    (batch_size, embedding_length) --> (batch_size,)
    (batch_size, 256) --> (batch_size, )
    '''
    def __init__(self, embedding_length=256, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.input_dim = embedding_length
        self.hidden_dim = hidden_dim
        layers = [
            nn.Linear(embedding_length, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ]
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, input_channel, output_channel, stride):
        super(DecoderBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1,output_padding=1),
            nn.BatchNorm1d(output_channel),
            Swish_act(),

            nn.ConvTranspose1d(output_channel, output_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.skip_connection = nn.Sequential()
        #if output_channel != input_channel:
        self.skip_connection = nn.Sequential(
            nn.ConvTranspose1d(input_channel, output_channel, kernel_size=1, stride=stride,output_padding=1),
            nn.BatchNorm1d(output_channel)
        )

        self.Lrelu = Swish_act()

    def forward(self, x):
        out = self.conv(x)
        #print(out.shape,x.shape,self.skip_connection(x).shape)
        out = self.skip_connection(x) + out
        out = self.Lrelu(out)
        return out

class Decoder(nn.Module):
    '''
    (batch_size, embedding_length) --> (batch_size, Channel, Seq_len)
    (batch_size, 256) --> (batch_size, C, 128)
    '''
    def __init__(self,embedding_length=256,output_channel=3):
        super(Decoder,self).__init__()
        self.embedding_length = embedding_length
        self.output_channel = output_channel

        self.linear = nn.Linear(embedding_length,1024)  # batch_size, 1, 1024  -> batch_szie,32,32

        self.layer1 = DecoderBlock(32,32,2)  # batch_size, 32, 64
        self.layer2 = DecoderBlock(32,32,2)  # batch_szie, 32,128

        self.conv1 = nn.Conv1d(32, output_channel,kernel_size=1)


    def forward(self,x):
        batch_size = x.shape[0]
        x = self.linear(x)
        x = x.view(batch_size,32,32)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.conv1(out)
        return out

class Predictor(nn.Module):
    '''
    (batch_size, embedding_length) --> (batch_size,)
    (batch_size, 256) --> (batch_size, )
    '''
    def __init__(self,embedding_length=256):
        super(Predictor,self).__init__()
        self.predit = nn.Sequential(
            nn.Linear(embedding_length,128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self,x):
        out = self.predit(x)
        return out

if __name__ == '__main__':
    batch_size,input_channel, seq_len = 16, 3, 256
    x = torch.randn(batch_size,seq_len)
    p = Decoder()
    y = p(x)
    print(y.shape)