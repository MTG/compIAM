from compiam.timbre.mridangam_stroke_classification import MridangamStrokeClassification

# Show user the available tools
import sys, inspect


def list_tools():
    list_of_tools = []
    for _, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            list_of_tools.append(obj.__name__)
    return list_of_tools
