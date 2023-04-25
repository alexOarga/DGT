from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix,
                                   to_undirected, to_dense_adj)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add

import numpy as np
import numba
import torch
import torch.nn as nn
from typing import Callable, List, Optional, Tuple, Union, Any

EPS = 1E-8


def compute_posenc_stats(data, pe_types, is_undirected, cfg):
    """Precompute positional encodings for the given graph.

    Supported PE statistics to precompute, selected by `pe_types`:
    'LapPE': Laplacian eigen-decomposition.
    'RWSE': Random walk landing probabilities (diagonals of RW matrices).
    'HKfullPE': Full heat kernels and their diagonals. (NOT IMPLEMENTED)
    'HKdiagSE': Diagonals of heat kernel diffusion.
    'ElstaticSE': Kernel based on the electrostatic interaction between nodes.

    Args:
        data: PyG graph
        pe_types: Positional encoding types to precompute statistics for.
            This can also be a combination, e.g. 'eigen+rw_landing'
        is_undirected: True if the graph is expected to be undirected
        cfg: Main configuration node

    Returns:
        Extended PyG Data object.
    """
    # Verify PE types.
    for t in pe_types:
        if t not in ['LapPE', 'MagLapPE', 'EquivStableLapPE', 'SignNet', 'RWSE', 'HKdiagSE', 'HKfullPE', 'ElstaticSE']:
            raise ValueError(f"Unexpected PE stats selection {t} in {pe_types}")

    # Basic preprocessing of the input graph.
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes  # Explicitly given number of nodes, e.g. ogbg-ppa
    else:
        N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
    laplacian_norm_type = cfg.posenc_LapPE.eigen.laplacian_norm.lower()
    if laplacian_norm_type == 'none':
        laplacian_norm_type = None
    if is_undirected:
        undir_edge_index = data.edge_index
    else:
        undir_edge_index = to_undirected(data.edge_index)

    # Eigen values and vectors.
    evals, evects = None, None
    if 'LapPE' in pe_types or 'EquivStableLapPE' in pe_types:
        # Eigen-decomposition with numpy, can be reused for Heat kernels.
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                           num_nodes=N)
        )
        evals, evects = np.linalg.eigh(L.toarray())
        
        if 'LapPE' in pe_types:
            max_freqs=cfg.posenc_LapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_LapPE.eigen.eigvec_norm
        elif 'EquivStableLapPE' in pe_types:  
            max_freqs=cfg.posenc_EquivStableLapPE.eigen.max_freqs
            eigvec_norm=cfg.posenc_EquivStableLapPE.eigen.eigvec_norm
        
        data.EigVals, data.EigVecs = get_lap_decomp_stats(
            evals=evals, evects=evects,
            max_freqs=max_freqs,
            eigvec_norm=eigvec_norm)

    if 'MagLapPE' in pe_types:
        if is_undirected:
            raise RuntimeError("Using magenitc laplacian with an undirected graph. This might be wrong.")

        eigenvalues, eigenvectors = aux_eigv_magnetic_laplacian_numba_parallel(
            senders=np.array(data.edge_index[0]),
            receivers=np.array(data.edge_index[1]),
            n_node=len(data.x),
            k=cfg.posenc_MagLapPE.eigen.max_freqs,
            k_excl=cfg.posenc_MagLapPE.excl_k_eigenvectors,
            q=cfg.posenc_MagLapPE.q,
            q_absolute=cfg.posenc_MagLapPE.q_absolute,
            norm_comps_sep=cfg.posenc_MagLapPE.norm_comps_sep,
            l2_norm=cfg.posenc_MagLapPE.l2_norm,
            sign_rotate=cfg.posenc_MagLapPE.sign_rotate,
            use_symmetric_norm=cfg.posenc_MagLapPE.symmetric_norm,
        )
        if len(eigenvalues) < cfg.posenc_MagLapPE.eigen.max_freqs:
            raise RuntimeError("Not enough eigenvalues found. Try reducing the number of eigenvalues to find.")

        eigenvalues = torch.tensor(eigenvalues, dtype=torch.float32)
        if np.iscomplexobj(eigenvectors):
            eigenvectors = torch.tensor(eigenvectors, dtype=torch.cfloat)
        else:
            eigenvectors = torch.tensor(eigenvectors, dtype=torch.float32)

        data.EigVals, data.EigVecs = eigenvalues, eigenvectors

    if 'SignNet' in pe_types:
        # Eigen-decomposition with numpy for SignNet.
        norm_type = cfg.posenc_SignNet.eigen.laplacian_norm.lower()
        if norm_type == 'none':
            norm_type = None
        L = to_scipy_sparse_matrix(
            *get_laplacian(undir_edge_index, normalization=norm_type,
                           num_nodes=N)
        )
        evals_sn, evects_sn = np.linalg.eigh(L.toarray())
        data.eigvals_sn, data.eigvecs_sn = get_lap_decomp_stats(
            evals=evals_sn, evects=evects_sn,
            max_freqs=cfg.posenc_SignNet.eigen.max_freqs,
            eigvec_norm=cfg.posenc_SignNet.eigen.eigvec_norm)

    # Random Walks.
    if 'RWSE' in pe_types:
        kernel_param = cfg.posenc_RWSE.kernel
        if len(kernel_param.times) == 0:
            raise ValueError("List of kernel times required for RWSE")
        rw_landing = get_rw_landing_probs(ksteps=kernel_param.times,
                                          edge_index=data.edge_index,
                                          num_nodes=N)
        data.pestat_RWSE = rw_landing

    # Heat Kernels.
    if 'HKdiagSE' in pe_types or 'HKfullPE' in pe_types:
        # Get the eigenvalues and eigenvectors of the regular Laplacian,
        # if they have not yet been computed for 'eigen'.
        if laplacian_norm_type is not None or evals is None or evects is None:
            L_heat = to_scipy_sparse_matrix(
                *get_laplacian(undir_edge_index, normalization=None, num_nodes=N)
            )
            evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        else:
            evals_heat, evects_heat = evals, evects
        evals_heat = torch.from_numpy(evals_heat)
        evects_heat = torch.from_numpy(evects_heat)

        # Get the full heat kernels.
        if 'HKfullPE' in pe_types:
            # The heat kernels can't be stored in the Data object without
            # additional padding because in PyG's collation of the graphs the
            # sizes of tensors must match except in dimension 0. Do this when
            # the full heat kernels are actually used downstream by an Encoder.
            raise NotImplementedError()
            # heat_kernels, hk_diag = get_heat_kernels(evects_heat, evals_heat,
            #                                   kernel_times=kernel_param.times)
            # data.pestat_HKdiagSE = hk_diag
        # Get heat kernel diagonals in more efficient way.
        if 'HKdiagSE' in pe_types:
            kernel_param = cfg.posenc_HKdiagSE.kernel
            if len(kernel_param.times) == 0:
                raise ValueError("Diffusion times are required for heat kernel")
            hk_diag = get_heat_kernels_diag(evects_heat, evals_heat,
                                            kernel_times=kernel_param.times,
                                            space_dim=0)
            data.pestat_HKdiagSE = hk_diag

    # Electrostatic interaction inspired kernel.
    if 'ElstaticSE' in pe_types:
        elstatic = get_electrostatic_function_encoding(undir_edge_index, N)
        data.pestat_ElstaticSE = elstatic

    return data


def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
    """Compute Laplacian eigen-decomposition-based PE stats of the given graph.

    Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs to use
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
    Returns:
        Tensor (num_nodes, max_freqs, 1) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
    """
    N = len(evals)  # Number of nodes, including disconnected nodes.

    # Keep up to the maximum desired number of frequencies.
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)

    # Normalize and pad eigen vectors.
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects

    # Pad and save eigenvalues.
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)

    return EigVals, EigVecs


def get_rw_landing_probs(ksteps, edge_index, edge_weight=None,
                         num_nodes=None, space_dim=0):
    """Compute Random Walk landing probabilities for given list of K steps.

    Args:
        ksteps: List of k-steps for which to compute the RW landings
        edge_index: PyG sparse representation of the graph
        edge_weight: (optional) Edge weights
        num_nodes: (optional) Number of nodes in the graph
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the random-walk diagonal by a factor `k^(space_dim/2)`.
            In euclidean space, this correction means that the height of
            the gaussian distribution stays almost constant across the number of
            steps, if `space_dim` is the dimension of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    source, dest = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
    deg_inv = deg.pow(-1.)
    deg_inv.masked_fill_(deg_inv == float('inf'), 0)

    if edge_index.numel() == 0:
        P = edge_index.new_zeros((1, num_nodes, num_nodes))
    else:
        # P = D^-1 * A
        P = torch.diag(deg_inv) @ to_dense_adj(edge_index, max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
    rws = []
    if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
        # Efficient way if ksteps are a consecutive sequence (most of the time the case)
        Pk = P.clone().detach().matrix_power(min(ksteps))
        for k in range(min(ksteps), max(ksteps) + 1):
            rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
            Pk = Pk @ P
    else:
        # Explicitly raising P to power k for each k \in ksteps.
        for k in ksteps:
            rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                       (k ** (space_dim / 2)))
    rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)

    return rw_landing


def get_heat_kernels_diag(evects, evals, kernel_times=[], space_dim=0):
    """Compute Heat kernel diagonal.

    This is a continuous function that represents a Gaussian in the Euclidean
    space, and is the solution to the diffusion equation.
    The random-walk diagonal should converge to this.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
        space_dim: (optional) Estimated dimensionality of the space. Used to
            correct the diffusion diagonal by a factor `t^(space_dim/2)`. In
            euclidean space, this correction means that the height of the
            gaussian stays constant across time, if `space_dim` is the dimension
            of the euclidean space.

    Returns:
        2D Tensor with shape (num_nodes, len(ksteps)) with RW landing probs
    """
    heat_kernels_diag = []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels diagonal only for each time
        eigvec_mul = evects ** 2
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
            this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                    dim=0, keepdim=False)

            # Multiply by `t` to stabilize the values, since the gaussian height
            # is proportional to `1/t`
            heat_kernels_diag.append(this_kernel * (t ** (space_dim / 2)))
        heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)

    return heat_kernels_diag


def get_heat_kernels(evects, evals, kernel_times=[]):
    """Compute full Heat diffusion kernels.

    Args:
        evects: Eigenvectors of the Laplacian matrix
        evals: Eigenvalues of the Laplacian matrix
        kernel_times: Time for the diffusion. Analogous to the k-steps in random
            walk. The time is equivalent to the variance of the kernel.
    """
    heat_kernels, rw_landing = [], []
    if len(kernel_times) > 0:
        evects = F.normalize(evects, p=2., dim=0)

        # Remove eigenvalues == 0 from the computation of the heat kernel
        idx_remove = evals < 1e-8
        evals = evals[~idx_remove]
        evects = evects[:, ~idx_remove]

        # Change the shapes for the computations
        evals = evals.unsqueeze(-1).unsqueeze(-1)  # lambda_{i, ..., ...}
        evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

        # Compute the heat kernels for each time
        eigvec_mul = (evects.unsqueeze(2) * evects.unsqueeze(1))  # (phi_{i, j1, ...} * phi_{i, ..., j2})
        for t in kernel_times:
            # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j1, ...} * phi_{i, ..., j2})
            heat_kernels.append(
                torch.sum(torch.exp(-t * evals) * eigvec_mul,
                          dim=0, keepdim=False)
            )

        heat_kernels = torch.stack(heat_kernels, dim=0)  # (Num kernel times) x (Num nodes) x (Num nodes)

        # Take the diagonal of each heat kernel,
        # i.e. the landing probability of each of the random walks
        rw_landing = torch.diagonal(heat_kernels, dim1=-2, dim2=-1).transpose(0, 1)  # (Num nodes) x (Num kernel times)

    return heat_kernels, rw_landing


def get_electrostatic_function_encoding(edge_index, num_nodes):
    """Kernel based on the electrostatic interaction between nodes.
    """
    L = to_scipy_sparse_matrix(
        *get_laplacian(edge_index, normalization=None, num_nodes=num_nodes)
    ).todense()
    L = torch.as_tensor(L)
    Dinv = torch.eye(L.shape[0]) * (L.diag() ** -1)
    A = deepcopy(L).abs()
    A.fill_diagonal_(0)
    DinvA = Dinv.matmul(A)

    electrostatic = torch.pinverse(L)
    electrostatic = electrostatic - electrostatic.diag()
    green_encoding = torch.stack([
        electrostatic.min(dim=0)[0],  # Min of Vi -> j
        electrostatic.max(dim=0)[0],  # Max of Vi -> j
        electrostatic.mean(dim=0),  # Mean of Vi -> j
        electrostatic.std(dim=0),  # Std of Vi -> j
        electrostatic.min(dim=1)[0],  # Min of Vj -> i
        electrostatic.max(dim=0)[0],  # Max of Vj -> i
        electrostatic.mean(dim=1),  # Mean of Vj -> i
        electrostatic.std(dim=1),  # Std of Vj -> i
        (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
        (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
    ], dim=1)

    return green_encoding


def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
    """
    Implement different eigenvector normalizations.
    """

    EigVals = EigVals.unsqueeze(0)

    if normalization == "L1":
        # L1 normalization: eigvec / sum(abs(eigvec))
        denom = EigVecs.norm(p=1, dim=0, keepdim=True)

    elif normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)

    elif normalization == "abs-max":
        # AbsMax normalization: eigvec / max|eigvec|
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values

    elif normalization == "wavelength":
        # AbsMax normalization, followed by wavelength multiplication:
        # eigvec * pi / (2 * max|eigvec| * sqrt(eigval))
        denom = torch.max(EigVecs.abs(), dim=0, keepdim=True).values
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom * 2 / np.pi

    elif normalization == "wavelength-asin":
        # AbsMax normalization, followed by arcsin and wavelength multiplication:
        # arcsin(eigvec / max|eigvec|)  /  sqrt(eigval)
        denom_temp = torch.max(EigVecs.abs(), dim=0, keepdim=True).values.clamp_min(eps).expand_as(EigVecs)
        EigVecs = torch.asin(EigVecs / denom_temp)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = eigval_denom

    elif normalization == "wavelength-soft":
        # AbsSoftmax normalization, followed by wavelength multiplication:
        # eigvec / (softmax|eigvec| * sqrt(eigval))
        denom = (F.softmax(EigVecs.abs(), dim=0) * EigVecs.abs()).sum(dim=0, keepdim=True)
        eigval_denom = torch.sqrt(EigVals)
        eigval_denom[EigVals < eps] = 1  # Problem with eigval = 0
        denom = denom * eigval_denom

    else:
        raise ValueError(f"Unsupported normalization `{normalization}`")

    denom = denom.clamp_min(eps).expand_as(EigVecs)
    EigVecs = EigVecs / denom

    return EigVecs


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
  n = n_node
  eigenvalues = np.zeros((k), dtype=np.float64)
  eigenvectors = np.zeros((n, k), dtype=np.complex128)

  padding_maks = senders >= 0

  eigenvalue, eigenvector, _ = aux_eigv_magnetic_laplacian_numba(
      senders[padding_maks],
      receivers[padding_maks],
      n_node,
      padded_nodes_size=n_node,
      k=k,
      k_excl=k_excl,
      q=q,
      q_absolute=q_absolute,
      norm_comps_sep=norm_comps_sep,
      l2_norm=l2_norm,
      sign_rotate=sign_rotate,
      use_symmetric_norm=use_symmetric_norm)

  eigenvalues[:eigenvalue.shape[0]] = eigenvalue
  eigenvectors[:eigenvector.shape[0], :eigenvector.shape[1]] = eigenvector
  return eigenvalues, eigenvectors
