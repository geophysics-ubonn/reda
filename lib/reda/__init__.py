rcParams = {}

# which inversion code should be used to compute geometric factors
# for now this will stay at 'crtomo'
rcParams['geom_factor.inversion_code'] = 'crtomo'

from .containers.ERT import ERT
from .containers.TDIP import TDIP
from .containers.CR import CR
from .containers.sEIT import sEIT
from .containers.SIP import SIP
from .testing import test
from .utils.helper_functions import search
import reda.utils.data as data
from .configs.configManager import ConfigManager

ERT
TDIP
CR
SIP
sEIT
test
search
data
ConfigManager
