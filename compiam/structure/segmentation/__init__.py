import sys

from compiam.utils import get_tool_list

from compiam.structure.segmentation.dhrupad_bandish_segmentation import (
    DhrupadBandishSegmentation,
)

# Show user the available tools
def list_tools():
    return get_tool_list(modules=sys.modules[__name__])
