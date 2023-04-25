import torch
import torch.nn.functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


@register_loss('weighted_cross_entropy')
def weighted_cross_entropy(pred, true):
    """Weighted cross-entropy for unbalanced classes.
    """
    if cfg.model.loss_fun == 'weighted_cross_entropy':
        # calculating label weights for weighted loss computation
        V = true.size(0)
        n_classes = pred.shape[1] if pred.ndim > 1 else 2
        label_count = torch.bincount(true, minlength=n_classes)
        # We dont filter non-zero classes
        #label_count = label_count[label_count.nonzero(as_tuple=True)].squeeze()
        label_count = label_count.squeeze()
        cluster_sizes = torch.zeros(n_classes, device=pred.device).long()
        #cluster_sizes[torch.unique(true)] = label_count
        cluster_sizes = label_count
        weight = (V - cluster_sizes).float() / V
        weight *= (cluster_sizes > 0).float()
        # multiclass
        if pred.ndim > 1:
            pred = F.log_softmax(pred, dim=-1)
            return F.nll_loss(pred, true, weight=weight), pred
        # binary
        else:
            #loss = F.binary_cross_entropy_with_logits(pred, true.float(),
            #                                          weight=weight[true])
            loss = F.binary_cross_entropy_with_logits(pred, true.float(),
                                                      weight=weight[true.type(torch.long)])
            return loss, torch.sigmoid(pred)
