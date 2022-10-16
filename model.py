import torch
import torch.nn as nn
from Nets import Encoder, Decoder, Predictor
from loss_fun import DiffLoss, AdversarialLoss, InfoNCE


class TransferNet(nn.Module):
    def __init__(self,args):
        super(TransferNet,self).__init__()
        self.args = args
        self.private_target_encoder = Encoder(input_channel=args.input_channel,embedding_length=args.embedding_length)
        self.private_source_encoder = Encoder(input_channel=args.input_channel,embedding_length=args.embedding_length)
        self.shared_encoder = Encoder(input_channel=args.input_channel,embedding_length=args.embedding_length)

        self.predictor = Predictor(embedding_length=args.embedding_length)
        self.shared_decoder = Decoder(embedding_length=args.embedding_length,output_channel=args.input_channel)

        self.similarity_loss = AdversarialLoss()
        self.difference_loss = DiffLoss()
        self.info_nce_loss = InfoNCE()

        self.predictor_loss = nn.MSELoss()

        if args.use_dsbn:
            self.source_domain_label = 'source'
            self.target_domain_label = 'target'
        else:
            self.source_domain_label = 'target'
            self.target_domain_label = 'target'

        self.predict_scheme = args.predict_scheme
        self.recon_scheme = args.recon_scheme

    def forward(self,source,target,source_label):
        # encoding process
        source_private_embedding = self.private_source_encoder(source,domain_label=self.source_domain_label)
        source_shared_embedding = self.shared_encoder(source,domain_label=self.source_domain_label)

        target_private_embedding = self.private_target_encoder(target,domain_label=self.target_domain_label)
        target_shared_embedding = self.shared_encoder(target,domain_label=self.target_domain_label)


        ########### L_difference
        source_diff_loss = self.difference_loss(source_private_embedding,source_shared_embedding)
        target_diff_loss = self.difference_loss(target_private_embedding,target_shared_embedding)
        diff_loss = source_diff_loss + target_diff_loss

        ########### L_similarity
        simi_loss = self.similarity_loss(source_shared_embedding,target_shared_embedding)


        ########### L_predictor
        if self.predict_scheme == 'share':
            source_pred_embedding = source_shared_embedding
        elif self.predict_scheme == 'all':
            source_pred_embedding = source_shared_embedding + source_private_embedding
        pred_label = self.predictor(source_pred_embedding)
        pred_loss = self.predictor_loss(source_label,pred_label)

        ########### L_info
        if self.recon_scheme == 'all':
            source_recon_embedding = source_shared_embedding + source_private_embedding
            target_recon_embedding = target_shared_embedding + target_private_embedding
        elif self.recon_scheme == 'share':
            source_recon_embedding = source_shared_embedding
            target_recon_embedding = target_shared_embedding
        source_reconstruction = self.shared_decoder(source_recon_embedding)
        target_reconstruction = self.shared_decoder(target_recon_embedding)

        bactch_size = source.shape[0]
        source = source.view(bactch_size,-1)
        target = target.view(bactch_size,-1)
        source_reconstruction = source_reconstruction.view(bactch_size,-1)
        target_reconstruction = target_reconstruction.view(bactch_size,-1)
        info_loss = self.info_nce_loss(source,source_reconstruction) + self.info_nce_loss(target,target_reconstruction)

        return diff_loss, simi_loss, info_loss, pred_loss

    def predict(self,x,domain_label='target'):
        target_private_embedding = self.private_target_encoder(x, domain_label=domain_label)
        target_shared_embedding = self.shared_encoder(x, domain_label=domain_label)

        if self.predict_scheme == 'share':
            target_pred_embedding = target_shared_embedding
        elif self.predict_scheme == 'all':
            target_pred_embedding = target_shared_embedding + target_private_embedding

        pred_label = self.predictor(target_pred_embedding)

        return pred_label

    def get_parameters(self,initial_lr=1.0):
        params = [
            {'params': self.private_target_encoder.parameters(), 'lr': initial_lr},
            {'params': self.private_source_encoder.parameters(), 'lr': initial_lr},
            {'params': self.shared_encoder.parameters(), 'lr': initial_lr},
            {'params': self.predictor.parameters(), 'lr': initial_lr},
            {'params': self.shared_decoder.parameters(), 'lr': initial_lr},
            {'params': self.similarity_loss.domain_classifier.parameters(), 'lr': initial_lr},
        ]
        return params

    def get_embedding(self, x, domain_label='target'):
        private_embedding = self.private_source_encoder(x, domain_label=domain_label)
        shared_embedding = self.shared_encoder(x, domain_label=domain_label)

        return private_embedding, shared_embedding

    def get_reconstruction(self, x, domain_label='target'):
        private_embedding = self.private_source_encoder(x, domain_label=domain_label)
        shared_embedding = self.shared_encoder(x, domain_label=domain_label)
        recon_embedding = private_embedding + shared_embedding
        reconstruction = self.shared_decoder(recon_embedding)
        return reconstruction



def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}



if __name__ == '__main__':
    import argparse
    def get_args():
        parser = argparse.ArgumentParser(description='test different length')
        parser.add_argument('--input_channel', default=6)
        parser.add_argument('--embedding_length', default=256)
        parser.add_argument('--use_dsbn', default=True)
        parser.add_argument('--predict_scheme', type=str, default='share', choices=['share', 'private', 'all'],)
        parser.add_argument('--recon_scheme', type=str, default='all',choices=['share', 'private', 'all'])
        args = parser.parse_args()
        return args
    args = get_args()

    source = torch.randn(16,args.input_channel,128)
    target = torch.randn(16,args.input_channel,128)
    source_label = torch.randn(16,1)

    net = TransferNet(args)

    l1,l2,l3,l4 = net(source,target,source_label)

    pred1 = net.predict(target,'t')

    print(l1,l2,l3,l4)

    print(pred1.shape)



    encoder = net.shared_encoder
    decoder = net.shared_decoder
    predictor = net.predictor

    num_encoder = get_parameter_number(encoder)
    num_decoder = get_parameter_number(decoder)
    num_predictor = get_parameter_number(predictor)

    total = 3*num_encoder['Trainable'] + num_decoder['Trainable'] + num_predictor['Trainable']

    test_num = num_encoder['Trainable'] + num_predictor['Trainable']
    print(test_num)

