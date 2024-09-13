


import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class SingleSrcNegSDR(_Loss):
    r"""Base class for single-source negative SI-SDR, SD-SDR and SNR.

    Args:
        sdr_type (str): choose between ``snr`` for plain SNR, ``sisdr`` for
            SI-SDR and ``sdsdr`` for SD-SDR [1].
        zero_mean (bool, optional): by default it zero mean the ref and
            estimate before computing the loss.
        take_log (bool, optional): by default the log10 of sdr is returned.
        reduction (string, optional): Specifies the reduction to apply to
            the output:
            ``'none'``: no reduction will be applied,
            ``'sum'``: the sum of the output
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output.
            
    Shape:
        - ests : :math:`(batch, time)`.
        - refs: :math:`(batch, time)`.

    Returns:
        :class:`torch.Tensor`: with shape :math:`(batch)` if ``reduction='none'`` else
        [] scalar if ``reduction='mean'``.

    Examples
        >>> import torch
        >>> from asteroid.losses import PITLossWrapper
        >>> refs = torch.randn(10, 2, 32000)
        >>> ests = torch.randn(10, 2, 32000)
        >>> loss_func = PITLossWrapper(SingleSrcNegSDR("sisdr"),
        >>>                            pit_from='pw_pt')
        >>> loss = loss_func(ests, refs)

    References
        [1] Le Roux, Jonathan, et al. "SDR half-baked or well done." IEEE
        International Conference on Acoustics, Speech and Signal
        Processing (ICASSP) 2019.

    Implementation modified from Astroid project: https://github.com/asteroid-team/asteroid
    """
    def __init__(self, 
                 sdr_type : str, 
                 zero_mean: bool = True, 
                 take_log: bool = True, 
                 reduction: str ="mean", 
                 EPS: float = 1e-8):
        super().__init__(reduction=reduction)

        assert sdr_type in ["snr", "sisdr", "sdsdr"]
        self.sdr_type = sdr_type
        self.zero_mean = zero_mean
        self.take_log = take_log
        self.EPS = 1e-8

    def forward(self, est, ref):

        assert est.shape == ref.shape, (
            'expected same shape, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        )
        assert len(est.shape) == len(ref.shape) == 3, (
            'expected BxCxN, get est: {}, ref: {} instead'.format(est.shape, ref.shape)
        )

        assert est.shape[1] == 1, (
            'expected mono channel, get {} channels'.format(est.shape[1])
        )

        B, C, T = est.shape
        est = est.reshape(-1, T)
        ref = ref.reshape(-1, T)

        # Step 1. Zero-mean norm
        if self.zero_mean:
            mean_source = torch.mean(ref, dim=1, keepdim=True)
            mean_estimate = torch.mean(est, dim=1, keepdim=True)
            ref = ref - mean_source
            est = est - mean_estimate
        # Step 2. Pair-wise SI-SDR.
        if self.sdr_type in ["sisdr", "sdsdr"]:
            # [batch, 1]
            dot = torch.sum(est * ref, dim=1, keepdim=True)
            # [batch, 1]
            s_ref_energy = torch.sum(ref**2, dim=1, keepdim=True) + self.EPS
            # [batch, time]
            scaled_ref = dot * ref / s_ref_energy
        else:
            # [batch, time]
            scaled_ref = ref
        if self.sdr_type in ["sdsdr", "snr"]:
            e_noise = est - ref
        else:
            e_noise = est - scaled_ref

        self.cache = torch.cat((ref, est, scaled_ref, e_noise), dim=0).detach().cpu()

        # [batch]
        losses = torch.sum(scaled_ref**2, dim=1) / (torch.sum(e_noise**2, dim=1) + self.EPS)

        if self.take_log:
            losses = 10 * torch.log10(losses + self.EPS)

        if self.reduction == "mean":
            losses = losses.mean()
        elif self.reduction == "sum":
            losses = losses.sum()
        else:
            losses = losses

        return -losses
    

SingleSISDRLoss = SingleSrcNegSDR("sisdr")
SingleSDSDRLoss = SingleSrcNegSDR("sdsdr")
SingleSNRLoss = SingleSrcNegSDR("snr")
