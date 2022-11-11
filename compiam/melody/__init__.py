### IMPORT HERE FUNCTIONALITIES
import inspect, importlib as implib

to_avoid = [
    x[0]
    for x in inspect.getmembers(
        implib.import_module("compiam.melody"), inspect.ismodule
    )
]


### IMPORT HERE THE CONSIDERED TASKS
from compiam.melody import tonic_identification
from compiam.melody import pitch_extraction
from compiam.melody import raga_recognition


# Show user the available tasks
def list_tasks():
    return [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.melody"), inspect.ismodule
        )
        if x[0] not in to_avoid
    ]


# Show user the available tools
def list_tools():
    tasks = [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.melody"), inspect.ismodule
        )
        if x[0] not in to_avoid
    ]
    tools_for_tasks = [
        inspect.getmembers(
            implib.import_module("compiam.melody." + task), inspect.isclass
        )
        for task in tasks
    ]
    return [task[0] for tasklist in tools_for_tasks for task in tasklist]
