from importlib.metadata import version

from queue_sim.queueSystem import *
import queue_sim.queueSystem as queueSystem

from queue_sim.server import *
import queue_sim.server as server

from queue_sim.policies import *
import queue_sim.policies as policies

from queue_sim.lib import *
import queue_sim.lib as lib

__version__ = version(__package__ or __name__)

__all__ = []
__all__.extend(["__version__"])
__all__.extend(queueSystem.__all__)
__all__.extend(server.__all__)
__all__.extend(policies.__all__)
__all__.extend(lib.__all__)