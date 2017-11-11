rcParams = {}

# which inversion code should be used to compute geometric factors
# for now this will stay at 'crtomo'
rcParams['geom_factor.inversion_code'] = 'crtomo'

from .containers.ERT import ERT
from .containers.sEIT import sEIT
from .testing import test
ERT
sEIT
