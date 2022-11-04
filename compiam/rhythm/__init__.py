import sys

from compiam.utils import get_tool_list

from compiam.rhythm.tabla_transcription import FourWayTabla
from compiam.rhythm.akshara_pulse_tracker import AksharaPulseTracker

# Show user the available tools
def list_tools():
    return get_tool_list(modules=sys.modules[__name__])
