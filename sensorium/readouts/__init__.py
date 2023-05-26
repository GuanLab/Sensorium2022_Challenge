
from .base import ClonedReadout, Readout
from .gaussian import (
    FullGaussian2d,
    RemappedGaussian2d,
)
from .multi_readout import MultiReadoutBase, MultiReadoutSharedParametersBase

### Userguide ###

# In order to build your multi-readout, pass the respective readout to your multi-readout base class
# together with your readout kwargs. Use the MultiReadoutSharedParametersBase if you want to share parameters
# between the readouts, otherwise use the MultiReadoutBase. Note that not all readouts support parameter sharing.

# Example:
# standard_multi_pointpooled_readout = MultiReadoutBase(PointPooled2d, **readout_kwargs)
# parameter_sharing_multi_gaussian_readout = MultiReadoutSharedParametersBase(FullGaussian2d, **readout_kwargs)
