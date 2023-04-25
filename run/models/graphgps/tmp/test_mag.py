import numpy as np
import numba
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union, Any

EPS = 1E-8


# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _norm_2d_along_first_dim_and_broadcast(array):
  """Equivalent to `linalg.norm(array, axis=0)[None, :] * ones_like(array)`."""
  output = np.zeros(array.shape, dtype=array.dtype)
  for i in numba.prange(array.shape[-1]):
    output[:, i] = np.linalg.norm(array[:, i])
  return output


# Necessary to work around numbas limitations with specifying axis in norm and
# braodcasting in parallel loops.
@numba.njit('float64[:, :](float64[:, :])', parallel=False)
def _max_2d_along_first_dim_and_broadcast(array):
  """Equivalent to `array.max(0)[None, :] * ones_like(array)`."""
  output = np.zeros(array.shape, dtype=array.dtype)
  for i in numba.prange(array.shape[-1]):
    output[:, i] = array[:, i].max()
  return output


#@numba.njit([
#    'Tuple((float64[::1], complex128[:, :], complex128[:, ::1]))(int64[:], ' +
#    'int64[:], int64[:], int64, int64, int64, float64, b1, b1, b1, b1, b1)'
#])
def aux_eigv_magnetic_laplacian_numba(
    senders: np.ndarray, receivers: np.ndarray, n_node: int,
    padded_nodes_size: int, k: int, k_excl: int, q: float, q_absolute: bool,
    norm_comps_sep: bool, l2_norm: bool, sign_rotate: bool,
    use_symmetric_norm: bool):
  """k non-ptrivial *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.
  Args:
    senders: Origin of the edges of shape [m].
    receivers: Target of the edges of shape [m].
    n_node: array shape [2]
    padded_nodes_size: int the number of nodes including padding.
    k: Returns top k eigenvectors.
    k_excl: The top (trivial) eigenvalues / -vectors to exclude.
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / m_imag / 2`.
    norm_comps_sep: If true first imaginary part is separately normalized.
    l2_norm: If true we use l2 normalization and otherwise the abs max value.
    sign_rotate: If true we decide on the sign based on max real values and
      rotate the imaginary part.
    use_symmetric_norm: symmetric (True) or row normalization (False).
  Returns:
    array of shape [<= k] containing the k eigenvalues.
    array of shape [n, <= k] containing the k eigenvectors.
    array of shape [n, n] the laplacian.
  """
  # Handle -1 padding
  edges_padding_mask = senders >= 0

  adj = np.zeros(int(padded_nodes_size * padded_nodes_size), dtype=np.float64)
  linear_index = receivers + (senders * padded_nodes_size).astype(senders.dtype)
  adj[linear_index] = edges_padding_mask.astype(adj.dtype)
  adj = adj.reshape(padded_nodes_size, padded_nodes_size)
  adj = np.where(adj > 1, 1, adj)

  symmetric_adj = adj + adj.T
  symmetric_adj = np.where((adj != 0) & (adj.T != 0), symmetric_adj / 2,
                           symmetric_adj)

  symmetric_deg = symmetric_adj.sum(-2)

  if not q_absolute:
    m_imag = (adj != adj.T).sum() / 2
    m_imag = min(m_imag, n_node)
    q = q / (m_imag if m_imag > 0 else 1)

  theta = 1j * 2 * np.pi * q * (adj - adj.T)

  if use_symmetric_norm:
    inv_deg = np.zeros((padded_nodes_size, padded_nodes_size), dtype=np.float64)
    np.fill_diagonal(
        inv_deg, 1. / np.sqrt(np.where(symmetric_deg < 1, 1, symmetric_deg)))
    eye = np.eye(padded_nodes_size)
    inv_deg = inv_deg.astype(adj.dtype)
    deg = inv_deg @ symmetric_adj.astype(adj.dtype) @ inv_deg
    laplacian = eye - deg * np.exp(theta)

    mask = np.arange(padded_nodes_size) < n_node
    mask = np.expand_dims(mask, -1) & np.expand_dims(mask, 0)
    laplacian = mask.astype(adj.dtype) * laplacian
  else:
    deg = np.zeros((padded_nodes_size, padded_nodes_size), dtype=np.float64)
    np.fill_diagonal(deg, symmetric_deg)
    laplacian = deg - symmetric_adj * np.exp(theta)

  eigenvalues, eigenvectors = np.linalg.eigh(laplacian)

  eigenvalues = eigenvalues[..., k_excl:k_excl + k]
  eigenvectors = eigenvectors[..., k_excl:k_excl + k]

  if sign_rotate:
    sign = np.zeros((eigenvectors.shape[1],), dtype=eigenvectors.dtype)
    for i in range(eigenvectors.shape[1]):
      argmax_i = np.abs(eigenvectors[:, i].real).argmax()
      sign[i] = np.sign(eigenvectors[argmax_i, i].real)
    eigenvectors = np.expand_dims(sign, 0) * eigenvectors

    argmax_imag_0 = eigenvectors[:, 0].imag.argmax()
    rotation = np.angle(eigenvectors[argmax_imag_0:argmax_imag_0 + 1])
    eigenvectors = eigenvectors * np.exp(-1j * rotation)

  if norm_comps_sep:
    # Only scale eigenvectors that seems to be more than numerical errors
    eps = EPS / np.sqrt(eigenvectors.shape[0])
    if l2_norm:
      scale_real = _norm_2d_along_first_dim_and_broadcast(np.real(eigenvectors))
      real = np.real(eigenvectors) / scale_real
    else:
      scale_real = _max_2d_along_first_dim_and_broadcast(
          np.abs(np.real(eigenvectors)))
      real = np.real(eigenvectors) / scale_real
    scale_mask = np.abs(
        np.real(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
    eigenvectors[:, scale_mask] = (
        real[:, scale_mask] + 1j * np.imag(eigenvectors)[:, scale_mask])

    if l2_norm:
      scale_imag = _norm_2d_along_first_dim_and_broadcast(np.imag(eigenvectors))
      imag = np.imag(eigenvectors) / scale_imag
    else:
      scale_imag = _max_2d_along_first_dim_and_broadcast(
          np.abs(np.imag(eigenvectors)))
      imag = np.imag(eigenvectors) / scale_imag
    scale_mask = np.abs(
        np.imag(eigenvectors)).sum(0) / eigenvectors.shape[0] > eps
    eigenvectors[:, scale_mask] = (
        np.real(eigenvectors)[:, scale_mask] + 1j * imag[:, scale_mask])
  elif not l2_norm:
    scale = _max_2d_along_first_dim_and_broadcast(np.absolute(eigenvectors))
    eigenvectors = eigenvectors / scale

  return eigenvalues.real, eigenvectors, laplacian


_eigv_magnetic_laplacian_numba_parallel_signature = [
    'Tuple((float64[:, :], complex128[:, :, :]))(int64[:, :], ' +
    'int64[:, :], int64[:, :], int64, int64, int64, float64, b1, b1, b1, b1, b1)'
]


def aux_eigv_magnetic_laplacian_numba_parallel(
    senders: np.ndarray,
    receivers: np.ndarray,
    n_node: np.ndarray,
    batch_size: int,
    k: int,
    k_excl: int,
    q: float,
    q_absolute: bool,
    norm_comps_sep: bool,
    l2_norm: bool,
    sign_rotate: bool,
    use_symmetric_norm: bool,
    # ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
):
  """k non-ptrivial *complex* eigenvectors of the smallest k eigenvectors of the magnetic laplacian.
  Args:
    senders: Origin of the edges of shape [b, m].
    receivers: Target of the edges of shape [b, m].
    n_node: array shape [b, 2]
    batch_size: batch size b.
    k: Returns top k eigenvectors.
    k_excl: The top (trivial) eigenvalues / -vectors to exclude.
    q: Factor in magnetic laplacian. Default 0.25.
    q_absolute: If true `q` will be used, otherwise `q / m_imag / 2`.
    norm_comps_sep: If true first imaginary part is separately normalized.
    l2_norm: If true we use l2 normalization and otherwise the abs max value.
      Will be treated as false if `norm_comps_sep` is true.
    sign_rotate: If true we decide on the sign based on max real values and
      rotate the imaginary part.
    use_symmetric_norm: symmetric (True) or row normalization (False).
  Returns:
    list with arrays of shape [<= k] containing the k eigenvalues.
    list with arrays of shape [n_i, <= k] containing the k eigenvectors.
  """
  n = n_node.sum(-1).max()
  eigenvalues = np.zeros((batch_size, k), dtype=np.float64)
  eigenvectors = np.zeros((batch_size, n, k), dtype=np.complex128)

  padding_maks = senders >= 0

  for i in range(0, batch_size, 1):
    eigenvalue, eigenvector, _ = aux_eigv_magnetic_laplacian_numba(
        senders[i][padding_maks[i]],
        receivers[i][padding_maks[i]],
        n_node[i],
        padded_nodes_size=n_node[i],
        k=k,
        k_excl=k_excl,
        q=q,
        q_absolute=q_absolute,
        norm_comps_sep=norm_comps_sep,
        l2_norm=l2_norm,
        sign_rotate=sign_rotate,
        use_symmetric_norm=use_symmetric_norm)

    eigenvalues[i, :eigenvalue.shape[0]] = eigenvalue
    eigenvectors[i, :eigenvector.shape[0], :eigenvector.shape[1]] = eigenvector
  return eigenvalues, eigenvectors

#############################################################################################

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 h_sizes,
                 activation=nn.ReLU,
                 with_norm=True,
                 final_activation=True):
        super(MLP, self).__init__()
        self.activation = activation
        self.with_norm = with_norm
        self.final_activation = final_activation
        self.hidden = nn.ModuleList()
        for i, size in enumerate(h_sizes):
            if i == 0:
                self.hidden.append(nn.LazyLinear(size))
            else:
                self.hidden.append(nn.LazyLinear(size))

    def forward(self, x):
        for layer in self.hidden:
            x = self.activation()(layer(x))
            if self.with_norm:
                x = nn.LayerNorm(x.shape[-1])(x)
        if self.final_activation is not None:
            x = self.activation()(x)
        return x

class MagLapNet(torch.nn.Module):
    def __init__(self,
                 d_model_elem: int = 32,
                 d_model_aggr: int = 256,
                 num_heads: int = 4,
                 n_layers: int = 1,
                 dropout_p: float = 0.2,
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU,
                 return_real_output: bool = True,
                 consider_im_part: bool = True,
                 use_signnet: bool = True,
                 use_gnn: bool = False,
                 use_attention: bool = False,
                 concatenate_eigenvalues: bool = False,
                 norm: Optional[Any] = None,
                 name: Optional[str] = None):
        super().__init__()
        self.concatenate_eigenvalues = concatenate_eigenvalues
        self.consider_im_part = consider_im_part
        self.use_signnet = use_signnet
        self.use_gnn = use_gnn
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.norm = norm

        if self.use_gnn:
            pass
        else:
            dim = int(2 * d_model_elem) if self.consider_im_part else d_model_elem
            self.element_mlp = MLP(
                input_dim=dim,
                h_sizes=[dim] * n_layers,
                activation=activation,
                with_norm=False,
                final_activation=True)

        self.re_aggregate_mlp = MLP(
            input_dim=d_model_aggr,
            h_sizes=[d_model_aggr] * n_layers,
            activation=activation,
            with_norm=False,
            final_activation=True)

        self.im_aggregate_mlp = None
        if not return_real_output and self.consider_im_part:
            self.im_aggregate_mlp = MLP(
                input_dim=d_model_aggr,
                h_sizes=[d_model_aggr] * n_layers,
                activation=activation,
                with_norm=False,
                final_activation=True)

    def forward(self,
                senders: torch.Tensor,
                receivers: torch.Tensor,
                eigenvalues: torch.Tensor,
                eigenvectors: torch.Tensor,):
        padding_mask = (eigenvalues > 0)[..., None, :]
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
                trans += self.element_mlp(-trans_eig)

        if self.concatenate_eigenvalues:
            eigenvalues_ = torch.broadcast_to(eigenvalues[..., None, :],
                                      trans.shape[:-1])
            trans = torch.stack((eigenvalues_[..., None], trans), dim=-1)

        if self.use_attention:
            if self.norm is not None:
                trans = self.norm()(trans)
            attn = torch.nn.MultiheadAttention(
                trans.shape[-1] // self.num_heads,
                self.num_heads,
                dropout=self.dropout_p,
                bias=False)
            trans += attn(
                trans,
                trans,
                trans,
                attn_mask=attn_padding_mask)

        padding_mask = padding_mask[..., None]
        trans = trans * padding_mask
        trans = trans.reshape(trans.shape[:-2] + (-1,))

        if self.dropout_p:
            trans = nn.Dropout(p=self.dropout_p)(trans)

        output = self.re_aggregate_mlp(trans)
        if self.im_aggregate_mlp is None:
            return output

        output_im = self.im_aggregate_mlp(trans)
        output = output + 1j * output_im
        return output

#############################################################################################

senders = []
receivers = []
weights = []
nodes = set()

NODES = 100

for i in range(NODES - 1):
    senders.append(i)
    receivers.append(i + 1)
    nodes.add(i)
    nodes.add(i + 1)
    weights.append(1)


eigenvalues, eigenvectors = aux_eigv_magnetic_laplacian_numba_parallel(
    senders = np.array([senders]),
    receivers = np.array([receivers]),
    n_node = np.array([len(nodes)]),
    batch_size = 1,
    k = 5,
    k_excl = 0,
    q = 0.25,
    q_absolute = False,
    norm_comps_sep = False,
    l2_norm = False,
    sign_rotate = True,
    use_symmetric_norm = False,
)

eigenvalues = torch.tensor(eigenvalues, dtype=torch.float32)
if np.iscomplexobj(eigenvectors):
    eigenvectors = torch.tensor(eigenvectors, dtype=torch.cfloat)
else:
    eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)

model = MagLapNet(
    d_model_elem = 32,
    d_model_aggr = 256,
    num_heads = 4,
    n_layers = 1,
    dropout_p = 0.2
)
out = model(
    torch.tensor(senders),
    torch.tensor(receivers),
    eigenvalues,
    eigenvectors)

val = eigenvectors[0]
import matplotlib.pyplot as plt
plt.plot(np.array(val[:, 0]), color='green')
plt.plot(np.array(val[:, 1]), color='blue')
plt.plot(np.array(val[:, 2]), color='yellow')
plt.plot(np.array(val[:, 3]), color='black')
plt.plot(np.array(val[:, 4]), color='green')
plt.ylabel('some numbers')
plt.show()

i = 0