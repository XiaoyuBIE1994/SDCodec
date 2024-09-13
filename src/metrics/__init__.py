
from torch.nn import L1Loss, MSELoss
from .sdr import (
    SingleSrcNegSDR,
)
from .spectrum import (
    MultiScaleSTFTLoss,
    MelSpectrogramLoss,
)

from .adv import (
    GANLoss,
)

from .visqol import VisqolMetric