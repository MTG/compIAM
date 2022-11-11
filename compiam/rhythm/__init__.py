### IMPORT HERE FUNCTIONALITIES
import inspect, importlib as implib

to_avoid = [
    x[0]
    for x in inspect.getmembers(
        implib.import_module("compiam.rhythm"), inspect.ismodule
    )
]


### IMPORT HERE THE CONSIDERED TASKS
from compiam.rhythm import meter_tracking
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
    tasks = [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.rhythm"), inspect.ismodule
        )
        if x[0] not in to_avoid
    ]
    tools_for_tasks = [
        inspect.getmembers(
            implib.import_module("compiam.rhythm." + task), inspect.isclass
        )
        for task in tasks
    ]
    return [task[0] for tasklist in tools_for_tasks for task in tasklist]
