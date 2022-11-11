import sys

from compiam.utils import get_tool_list

from compiam.melody.pitch_extraction.melodia import Melodia
from compiam.melody.pitch_extraction.ftanet_carnatic import FTANetCarnatic

# Show user the available tools
def list_tools():
    return get_tool_list(modules=sys.modules[__name__])
