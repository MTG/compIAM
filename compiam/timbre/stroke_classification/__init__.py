import sys

from compiam.utils import get_tool_list

# Import tasks
from compiam.timbre.stroke_classification.mridangam_stroke_classification import MridangamStrokeClassification

# Show user the available tools
def list_tools():
    return get_tool_list(modules=sys.modules[__name__])
