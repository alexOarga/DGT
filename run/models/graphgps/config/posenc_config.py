from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


@register_config('posenc')
def set_cfg_posenc(cfg):
    """Extend configuration with positional encoding options.
    """

    # Argument group for each Positional Encoding class.
    cfg.posenc_LapPE = CN()
    cfg.posenc_SignNet = CN()
    cfg.posenc_RWSE = CN()
    cfg.posenc_HKdiagSE = CN()
    cfg.posenc_ElstaticSE = CN()
    cfg.posenc_EquivStableLapPE = CN()
    cfg.posenc_MagLapPE = CN()

    # Common arguments to all PE types.
    for name in ['posenc_LapPE', 'posenc_SignNet',
                 'posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE', 'posenc_MagLapPE']:
        pecfg = getattr(cfg, name)

        # Use extended positional encodings
        pecfg.enable = False

        # Neural-net model type within the PE encoder:
        # 'DeepSet', 'Transformer', 'Linear', 'none', ...
        pecfg.model = 'none'

        # Size of Positional Encoding embedding
        pecfg.dim_pe = 16

        # Number of layers in PE encoder model
        pecfg.layers = 3

        # Number of attention heads in PE encoder when model == 'Transformer'
        pecfg.n_heads = 4

        # Number of layers to apply in LapPE encoder post its pooling stage
        pecfg.post_layers = 0

        # Choice of normalization applied to raw PE stats: 'none', 'BatchNorm'
        pecfg.raw_norm_type = 'none'

        # In addition to appending PE to the node features, pass them also as
        # a separate variable in the PyG graph batch object.
        pecfg.pass_as_var = False

    # Config for EquivStable LapPE
    cfg.posenc_EquivStableLapPE.enable = False
    cfg.posenc_EquivStableLapPE.raw_norm_type = 'none'

    # Config for Laplacian Eigen-decomposition for PEs that use it.
    for name in ['posenc_LapPE', 'posenc_SignNet', 'posenc_EquivStableLapPE', 'posenc_MagLapPE']:
        pecfg = getattr(cfg, name)
        pecfg.eigen = CN()

        # The normalization scheme for the graph Laplacian: 'none', 'sym', or 'rw'
        pecfg.eigen.laplacian_norm = 'sym'

        # The normalization scheme for the eigen vectors of the Laplacian
        pecfg.eigen.eigvec_norm = 'L2'

        # Maximum number of top smallest frequencies & eigenvectors to use
        pecfg.eigen.max_freqs = 10

    # Config for SignNet-specific options.
    cfg.posenc_SignNet.phi_out_dim = 4
    cfg.posenc_SignNet.phi_hidden_dim = 64

    for name in ['posenc_RWSE', 'posenc_HKdiagSE', 'posenc_ElstaticSE']:
        pecfg = getattr(cfg, name)

        # Config for Kernel-based PE specific options.
        pecfg.kernel = CN()

        # List of times to compute the heat kernel for (the time is equivalent to
        # the variance of the kernel) / the number of steps for random walk kernel
        # Can be overridden by `posenc.kernel.times_func`
        pecfg.kernel.times = []

        # Python snippet to generate `posenc.kernel.times`, e.g. 'range(1, 17)'
        # If set, it will be executed via `eval()` and override posenc.kernel.times
        pecfg.kernel.times_func = ''

    # Override default, electrostatic kernel has fixed set of 10 measures.
    cfg.posenc_ElstaticSE.kernel.times_func = 'range(10)'

    # Config for magentic eigv.
    cfg.posenc_MagLapPE.excl_k_eigenvectors = 1
    cfg.posenc_MagLapPE.q = 0.25
    cfg.posenc_MagLapPE.q_absolute = True
    cfg.posenc_MagLapPE.symmetric_norm = False
    cfg.posenc_MagLapPE.norm_comps_sep = False
    cfg.posenc_MagLapPE.l2_norm = True
    cfg.posenc_MagLapPE.sign_rotate = True

    # Config for MagLapNet
    cfg.posenc_MagLapPE.d_model_elem = 32
    cfg.posenc_MagLapPE.d_model_aggr = 256
    cfg.posenc_MagLapPE.num_heads = 4
    cfg.posenc_MagLapPE.n_layers = 1
    cfg.posenc_MagLapPE.dropout_p = 0.2
    cfg.posenc_MagLapPE.return_real_output = True
    cfg.posenc_MagLapPE.consider_im_part = True
    cfg.posenc_MagLapPE.use_signnet = True
    cfg.posenc_MagLapPE.use_gnn = False
    cfg.posenc_MagLapPE.use_attention = False
    cfg.posenc_MagLapPE.concatenate_eigenvalues = False
    cfg.posenc_MagLapPE.norm = None