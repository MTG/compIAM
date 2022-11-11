import os
import pathlib

WORKDIR = os.path.dirname(pathlib.Path(__file__).parent.resolve())

#############
# Models Dict
#############
# Each model should be stored in models_dict using
#   "<melody|rhythm|timbrestructure>:model_id":<d> where <d> is:
# 	    {
#           "module_name": "<compiam.melody|rhythm|timbre|structure.task.file|folder name of model>",
#           "class_name": "<name of the model class>",
#           "model_path": "<path_to_model if any>"
#       }

models_dict = {
    "melody:deepsrgm": {
        "module_name": "compiam.melody.raga_recognition.deepsrgm",
        "class_name": "DEEPSRGM",
        "kwargs": {
            "model_path": os.path.join(WORKDIR, "models", "melody", "deepsrgm"),
            "mapping_path": os.path.join(
                WORKDIR,
                "compiam",
                "melody",
                "deepsrgm",
                "conf",
                "DEEPSRGM_mapping.json",
            ),
        },
    },
    "melody:ftanet-carnatic": {
        "module_name": "compiam.melody.pitch_extraction.ftanet_carnatic",
        "class_name": "FTANetCarnatic",
        "kwargs": {
            "model_path": os.path.join(
                WORKDIR, "models", "melody", "ftanet", "carnatic", "OA"
            )
        },
    },
    "rhythm:1way-tabla": {
        "module_name": "compiam.rhythm.transcription.tabla_transcription",
        "class_name": "FourWayTabla",
        "kwargs": {
            "model_path": os.path.join(WORKDIR, "models", "rhythm", "4wayTabla", "1way")
        },
    },
    "rhythm:4way-tabla": {
        "module_name": "compiam.rhythm.transcription.tabla_transcription",
        "class_name": "FourWayTabla",
        "kwargs": {
            "model_path": os.path.join(WORKDIR, "models", "rhythm", "4wayTabla", "4way")
        },
    },
    "structure:dhrupad-bandish-segmentation": {
        "module_name": "compiam.structure.segmentation.dhrupad_bandish_segmentation",
        "class_name": "DhrupadBandishSegmentation",
        "kwargs": {
            "model_path": os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "pretrained_models",
            ),
            "splits_path": os.path.join(
                WORKDIR, "models", "structure", "dhrupad_bandish_segmentation", "splits"
            ),
            "annotations_path": os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "annotations",
            ),
            "original_audios_path": os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "audio_original",
            ),
            "processed_audios_path": os.path.join(
                WORKDIR,
                "models",
                "structure",
                "dhrupad_bandish_segmentation",
                "audio_sections",
            ),
        },
    },
}


###############
# Datasets List
###############
# Make sure you:
#   1. create a dataset loader in mirdata (https://github.com/mir-dataset-loaders/mirdata)
#   2. add the dataset identifier in this list

datasets_list = ["saraga_carnatic", "saraga_hindustani", "mridangam_stroke"]


##############
# Corpora List
##############
# Do not edit these dict: it is fixed in Dunya and no additional corpus is scheduled at the moment.

corpora_list = {
    "carnatic": {
        "dunya-carnatic": {
            "name": "Dunya Carnatic",
            "description": "Dunya Carnatic",
            "slug": "dunya-carnatic",
            "root_directory": "/incoming/Carnatic/",
            "id": 7,
        },
        "dunya-carnatic-cc": {
            "name": "Dunya Carnatic CC",
            "description": "Dunya CC collection in musicbrainz (basically the same as Saraga)",
            "slug": "dunya-carnatic-cc",
            "root_directory": "/incoming/CarnaticCC",
            "id": 18,
        },
    },
    "hindustani": {
        "dunya-hindustani-cc": {
            "name": "Dunya Hindustani CC",
            "description": "Creative commons licensed Hindustani music",
            "slug": "dunya-hindustani-cc",
            "root_directory": "/incoming/HindustaniCC",
            "id": 19,
        },
        "dunya-hindustani": {
            "name": "Dunya Hindustani",
            "description": "Commercial Hindustani recordings",
            "slug": "dunya-hindustani",
            "root_directory": "/incoming/Hindustani",
            "id": 15,
        },
    },
}
