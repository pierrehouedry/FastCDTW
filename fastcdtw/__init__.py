"""FastCDTW: time warping by curve length, squared Euclidean cost (p = 2)."""

from .cdist_fit import as_tensor, cdist_fm_dtw_fit
from .interp import (dedup_times, interp, interp_batch, interp_inverse,
                     interp_series)
from .loss import (eval_warping, fm_dtw, fm_dtw_matrix, fm_dtw_mc,
                   sample_times, seg_integral)
from .loss_shared_time import breakpoints, cdist_fm_dtw_mc, cdist_fm_dtw_ref
from .loss_shared_time_exact import MAD_FMTW, cdist_fm_dtw
from .warping import CumSumWarping, INRWarping, phi_from_raw

__all__ = [
    "interp", "interp_series", "interp_batch", "interp_inverse", "dedup_times",
    "seg_integral", "fm_dtw", "fm_dtw_matrix", "fm_dtw_mc",
    "sample_times", "eval_warping",
    "breakpoints", "cdist_fm_dtw", "cdist_fm_dtw_ref", "cdist_fm_dtw_mc",
    "cdist_fm_dtw_fit", "as_tensor",
    "MAD_FMTW",
    "phi_from_raw", "CumSumWarping", "INRWarping",
]
