import warnings

import torch
import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_node_encoder
from torchvision.ops import MLP
from performer_pytorch import SelfAttention, Performer

EPS = 1E-8

'''
class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 h_sizes,
                 #activation=nn.ReLU,
                 with_norm=True,
                 final_activation=True):
        super(MLP, self).__init__()
        #self.activation = activation
        self.with_norm = with_norm
        #self.final_activation = final_activation
        self.hidden = nn.ModuleList()
        for i, size in enumerate(h_sizes):
            if i == 0:
                self.hidden.append(nn.Linear(input_dim, size))
                self.hidden.append(nn.ReLU())
            else:
                self.hidden.append(nn.Linear(h_sizes[i - 1], size))
                self.hidden.append(nn.ReLU())
            #if self.with_norm:
            #    self.hidden.append(nn.LayerNorm(size))
        #if self.final_activation:
        #    self.hidden.append(nn.ReLU())

    def forward(self, x):
        for layer in self.hidden:
            x = layer(x)
        return x
'''
@register_node_encoder('MagLapNet')
class MagLapNet(torch.nn.Module):
    def __init__(self,
                 dim_emb,
                 expand_x=True,
                 activation = nn.ReLU):
        super(MagLapNet, self).__init__()

        d_model_elem = cfg.posenc_MagLapPE.d_model_elem
        d_model_aggr = cfg.posenc_MagLapPE.d_model_aggr
        num_heads = cfg.posenc_MagLapPE.num_heads
        n_layers = cfg.posenc_MagLapPE.n_layers
        dropout_p = cfg.posenc_MagLapPE.dropout_p
        return_real_output = cfg.posenc_MagLapPE.return_real_output
        consider_im_part = cfg.posenc_MagLapPE.consider_im_part
        use_signnet = cfg.posenc_MagLapPE.use_signnet
        use_gnn = cfg.posenc_MagLapPE.use_gnn
        use_attention = cfg.posenc_MagLapPE.use_attention
        concatenate_eigenvalues= cfg.posenc_MagLapPE.concatenate_eigenvalues
        norm = cfg.posenc_MagLapPE.norm
        max_freqs = cfg.posenc_MagLapPE.eigen.max_freqs

        self.concatenate_eigenvalues = concatenate_eigenvalues
        self.consider_im_part = consider_im_part
        self.use_signnet = use_signnet
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.norm = norm

        if dim_emb - d_model_aggr < 1:
            warnings.warn(f"LapPE size {d_model_aggr} is too large for "
                             f"desired embedding size of {dim_emb}.")

        if expand_x:
            self.linear_x = nn.Linear(d_model_aggr, dim_emb - d_model_aggr)
        self.expand_x = expand_x

        if self.use_gnn:
            pass
        else:
            dim = int(2 * d_model_elem) if self.consider_im_part else d_model_elem
            '''
            self.element_mlp = MLP(
                input_dim=2,
                h_sizes=[dim] * n_layers,
                #activation=activation,
                with_norm=False,
                final_activation=True)

            self.element_mlp_neg = MLP(
                input_dim=2,
                h_sizes=[dim] * n_layers,
                # activation=activation,
                with_norm=False,
                final_activation=True)
            '''
            self.element_mlp = MLP(
                2,
                [dim] * n_layers)

        '''
        self.re_aggregate_mlp = MLP(
            input_dim=2 * d_model_elem * max_freqs,
            h_sizes=[d_model_aggr] * n_layers,
            #activation=activation,
            with_norm=False,
            final_activation=True)

        self.im_aggregate_mlp = None
        if not return_real_output and self.consider_im_part:
            self.im_aggregate_mlp = MLP(
                input_dim=2 * d_model_elem * max_freqs,
                h_sizes=[d_model_aggr] * n_layers,
                #activation=activation,
                with_norm=False,
                final_activation=True)

        '''

        self.re_aggregate_mlp = MLP(
            2 * d_model_elem * max_freqs,
            [d_model_aggr] * n_layers)

        self.im_aggregate_mlp = None
        if not return_real_output and self.consider_im_part:
            self.im_aggregate_mlp = MLP(
                2 * d_model_elem * max_freqs,
                [d_model_aggr] * n_layers)

        if use_attention:
            '''
            encoder_layer = nn.TransformerEncoderLayer(d_model=2 * d_model_elem,
                                                       nhead=num_heads,
                                                       batch_first=True)
            self.attn = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=n_layers)
            '''
            self.attn = SelfAttention(
                dim=2 * d_model_elem, heads=num_heads,
                dropout=self.dropout_p, causal=False)

    def forward(self,
                batch):
        if not (hasattr(batch, 'EigVals') and hasattr(batch, 'EigVecs')):
            raise ValueError("Precomputed eigen values and vectors are "
                             f"required for {self.__class__.__name__}; "
                             "set config 'posenc_LapPE.enable' to True")
        senders = batch.edge_index[0]
        receivers = batch.edge_index[1]
        eigenvalues = batch.EigVals
        eigenvectors = batch.EigVecs

        #eigenvalues = eigenvalues[None, :]
        #eigenvectors = eigenvectors[None, :]

        #padding_mask = (eigenvalues > 0)[..., None, :]
        padding_mask = (eigenvalues > 0)
        padding_mask[..., 0] = True
        attn_padding_mask = padding_mask[..., None] & padding_mask[..., None, :]

        trans_eig = eigenvectors.real
        trans_eig = trans_eig[..., None]

        if self.consider_im_part and torch.is_complex(eigenvectors):
            trans_eig_im = eigenvectors.imag[..., None]
            trans_eig = torch.cat((trans_eig, trans_eig_im), dim=-1)

        if self.use_gnn:
            pass
        else:
            trans = self.element_mlp(trans_eig)
            if self.use_signnet:
                aux = self.element_mlp(-trans_eig)
                trans = trans + aux

        if self.concatenate_eigenvalues:
            eigenvalues_ = torch.broadcast_to(eigenvalues[..., None, :],
                                      trans.shape[:-1])
            trans = torch.stack((eigenvalues_[..., None], trans), dim=-1)

        if self.use_attention:
            if self.norm is not None:
                trans = self.norm()(trans)
            empty_mask = torch.isnan(trans)
            '''
            attn_output = self.attn(src=trans,
                                      src_key_padding_mask=empty_mask[:, :, 0])
            '''
            attn_output = self.attn(trans)
            trans = trans + attn_output

        padding_mask = padding_mask[..., None]
        #trans = trans * padding_mask
        trans = trans.reshape(trans.shape[:-2] + (-1,))

        if self.dropout_p:
            trans = nn.Dropout(p=self.dropout_p)(trans)

        output = self.re_aggregate_mlp(trans)

        if self.im_aggregate_mlp is None:
            pass
        else:
            output_im = self.im_aggregate_mlp(trans)
            output = output + 1j * output_im

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        #batch.x = torch.cat((h, output), 1)
        batch.x = h + output

        return batch