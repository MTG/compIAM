# Pitch extraction
from compiam.melody.melodia import Melodia
from compiam.melody.ftanet_carnatic import FTANetCarnatic

# Tonic extraction
from compiam.melody.tonic_multipitch import TonicIndianMultiPitch

# Raga recognition
from compiam.melody.deepsrgm import DEEPSRGM

# Show user the available tools
import sys, inspect
def list_tools():
    list_of_tools = []
    for _, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            list_of_tools.append(obj.__name__)
    return list_of_tools