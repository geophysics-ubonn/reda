


from reda.utils.enter_directory import CreateEnterDirectory
from reda.utils.enter_directory import EnterDirectory
import reda.importers.utils.transforms as transforms
from .configs.configManager import ConfigManager
import reda.utils.data as data
from .utils.helper_functions import search
from .container.BaseContainer import BaseContainer
from .containers.SIP import SIP
from .containers.sEIT import sEIT
from .containers.CR import CR
from .containers.TDIP import TDIP
from .containers.ERT import ERT
from .filter.DCA import DCA
from .testing import test
rcParams = {}

# which inversion code should be used to compute geometric factors
# for now this will stay at 'crtomo'
rcParams['geom_factor.inversion_code'] = 'crtomo'


# this is just to silence pep8/flake errors
EnterDirectory
CreateEnterDirectory
BaseContainer
ERT
DCA
TDIP
CR
SIP
sEIT
test
search
data
ConfigManager
transforms
