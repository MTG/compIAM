import sys

from compiam.utils import get_tool_list

from compiam.melody.raga_recognition.deepsrgm import DEEPSRGM

# Show user the available tools
def list_tools():
    return get_tool_list(modules=sys.modules[__name__])