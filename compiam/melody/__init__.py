# Pitch extraction
from compiam.melody.melodia import Melodia
from compiam.melody.ftanet_carnatic import FTANetCarnatic

# Tonic extraction
from compiam.melody.tonic_multipitch import TonicIndianMultiPitch

# Raga recognition
from compiam.melody.deepsrgm import DEEPSRGM

# Show user the available tools
def list_tools():
    return ["Melodia", "FTANetCarnatic", "TonicIndianMultiPitch", "DEEPSRGM"]
