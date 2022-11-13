import sys

from compiam.utils import get_tool_list
from compiam.data import models_dict

from compiam.rhythm.transcription.tabla_transcription import FourWayTabla
from compiam.rhythm.transcription.mnemonic_transcription import MnemonicTranscription

# Show user the available tools
def list_tools():
    pre_trained_models = [
        x["class_name"] for x in list(models_dict.values())
    ]  # Get list of pre-trained_models
    return [
        tool + "*" if tool in pre_trained_models else tool
        for tool in get_tool_list(modules=sys.modules[__name__])
    ]
