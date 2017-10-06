rcParams = {}

# which inversion code should be used to compute geometric factors
# for now this will stay at 'crtomo'
rcParams['geom_factor.inversion_code'] = 'crtomo'

from edf.containers.ERT import ERT
from edf.containers.sEIT import sEIT
ERT
sEIT
