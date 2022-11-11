### IMPORT HERE FUNCTIONALITIES
import inspect, importlib as implib

to_avoid = [
    x[0]
    for x in inspect.getmembers(
        implib.import_module("compiam.timbre"), inspect.ismodule
    )
]


### IMPORT HERE THE CONSIDERED TASKS
from compiam.timbre import stroke_classification


# Show user the available tasks
def list_tasks():
    return [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.timbre"), inspect.ismodule
        )
        if x[0] not in to_avoid
    ]


# Show user the available tools
def list_tools():
    tasks = [
        x[0]
        for x in inspect.getmembers(
            implib.import_module("compiam.timbre"), inspect.ismodule
        )
        if x[0] not in to_avoid
    ]
    return [
        inspect.getmembers(
            implib.import_module("compiam.timbre." + task), inspect.isclass
        )[0][0]
        for task in tasks
    ]
