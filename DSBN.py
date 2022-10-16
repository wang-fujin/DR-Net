import torch
import torch.nn as nn


class DomainSpecificBatchNorm1D(nn.Module):
    _version = 2

    def __init__(self,num_features, num_domains=2, eps=1e-5,momentum=0.1, affine=True, track_running_stats=True):
        super(DomainSpecificBatchNorm1D,self).__init__()
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(num_features,eps,momentum,affine,track_running_stats) for _ in range(num_domains)]
        )

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _chect_input_dim(self,input):
        if input.dim() != 3:
            raise ValueError('expected 3D input, but got {}D input'.format(input.dim()))

    def forward(self, x, domain_label):
        self._chect_input_dim(x)
        if domain_label == 'source' or domain_label=='s':
            bn = self.bns[0]
        elif domain_label == 'target' or domain_label=='t':
            bn = self.bns[1]
        else :
            raise ValueError('"domain label" must be "source/s" or "target/t", but got "{}".'.format(domain_label))
        return bn(x)

if __name__ == "__main__":
    xs = torch.randn(16,32,100) # batch_size, channel, seq_len
    xt = torch.randn(16,32,100)
    dsbn = DomainSpecificBatchNorm1D(num_features=32)
    y1 = dsbn(xs,'source')
    y2 = dsbn(xt,'target')
    print(dsbn)

