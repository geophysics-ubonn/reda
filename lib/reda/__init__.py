import reda.main.logger
from reda.utils.enter_directory import CreateEnterDirectory
from reda.utils.enter_directory import EnterDirectory
import reda.importers.utils.transforms as transforms
from .configs.configManager import ConfigManager
import reda.utils.data as data
from .utils.helper_functions import search
from .containers.BaseContainer import BaseContainer
from .containers.SIP import SIP
from .containers.sEIT import sEIT
from .containers.CR import CR
from .containers.TDIP import TDIP
from .containers.ERT import ERT
from .testing import test
import reda.importers.eit_fzj as eit_fzj
from reda.importers.fzj_readbin import fzj_readbin
from reda.utils.electrode_manager import electrode_manager
rcParams = {}

# which inversion code should be used to compute geometric factors
# for now this will stay at 'crtomo'
rcParams['geom_factor.inversion_code'] = 'crtomo'

# this is just to silence pep8/flake errors
reda.main.logger
EnterDirectory
CreateEnterDirectory
BaseContainer
ERT
TDIP
CR
SIP
sEIT
test
search
data
ConfigManager
transforms
eit_fzj
fzj_readbin
electrode_manager


def version():
    """Return the installed version of reda, using the pkg_resources package"""
    import pkg_resources
    return pkg_resources.require('reda')[0].version
