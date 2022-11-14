### IMPORT HERE FUNCTIONALITIES
import inspect, importlib as implib
from compiam.data import models_dict

to_avoid = [
    x[0]
    for x in inspect.getmembers(
        implib.import_module("compiam.rhythm"), inspect.ismodule
    )
]


### IMPORT HERE THE CONSIDERED TASKS
from compiam.rhythm import meter
from compiam.rhythm import transcription


# Show user the available tasks
def list_tasks():
    return [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.rhythm"), inspect.ismodule
        )
        if x[0] not in to_avoid
    ]


# Show user the available tools
def list_tools():
    tools = [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.rhythm"), inspect.ismodule
        )
        if x[0] not in to_avoid
    ]
    tools_for_tasks = [
        inspect.getmembers(
            implib.import_module("compiam.rhythm." + tool), inspect.isclass
        )
        for tool in tools
    ]
    tools_for_tasks = [
        tool[1].__module__.split(".")[-2] + "." + tool[0]
        for tool_list in tools_for_tasks
        for tool in tool_list
    ]  # Get task.tool
    pre_trained_models = [
        x["class_name"] for x in list(models_dict.values())
    ]  # Get list of pre-trained_models
    return [
        tool + "*" if tool.split(".")[1] in pre_trained_models else tool
        for tool in tools_for_tasks
    ]
