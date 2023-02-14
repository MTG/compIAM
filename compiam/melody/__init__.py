### IMPORT HERE FUNCTIONALITIES
import inspect, importlib as implib
from compiam.data import models_dict

TO_AVOID = [
    x[0]
    for x in inspect.getmembers(
        implib.import_module("compiam.melody"), inspect.ismodule
    )
]


### IMPORT HERE THE CONSIDERED TASKS
from compiam.melody import tonic_identification
from compiam.melody import pitch_extraction
from compiam.melody import raga_recognition
from compiam.melody import pattern


# Show user the available tasks
def list_tasks():
    return [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.melody"), inspect.ismodule
        )
        if x[0] not in TO_AVOID
    ]


# Show user the available tools
def list_tools():
    tasks = [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.melody"), inspect.ismodule
        )
        if x[0] not in TO_AVOID
    ]
    tools_for_tasks = [
        inspect.getmembers(
            implib.import_module("compiam.melody." + task), inspect.isclass
        )
        for task in tasks
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
