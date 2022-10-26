import os
import pathlib

WORKDIR = os.path.dirname(pathlib.Path(__file__).parent.resolve())

#############
# Models Dict
#############
# Each model should be stored in models_dict using
#   "<melody/rhythm/timbre/structure>:model_id":<d> where <d> is:
# 	    {
#           "module_name": "<compiam.melody/rhythm/timbre/structure.file/folder name of model>",
#           "class_name": "<name of the model class>",
#           "model_path": "<path_to_model if any>"
#       }

models_dict = {
    "rhythm:1way-tabla": {
        "module_name": "compiam.rhythm.tabla_transcription",
        "class_name": "FourWayTabla",
        "kwargs": {
            "model_path": os.path.join(WORKDIR, "models/rhythm/4wayTabla/1way/")
        },
    },
    "rhythm:4way-tabla": {
        "module_name": "compiam.rhythm.tabla_transcription",
        "class_name": "FourWayTabla",
        "kwargs": {
            "model_path": os.path.join(WORKDIR, "models/rhythm/4wayTabla/4way/")
        },
    },
    "melody:ftanet-carnatic": {
        "module_name": "compiam.melody.ftanet_carnatic",
        "class_name": "FTANetCarnatic",
        "kwargs": {
            "model_path": os.path.join(WORKDIR, "models/melody/ftanet/carnatic/OA")
        },
    },
    "melody:melodia": {
        "module_name": "compiam.melody.melodia",
        "class_name": "Melodia",
        "kwargs": {},
    },
    "melody:tonic-multipitch": {
        "module_name": "compiam.melody.tonic_multipitch",
        "class_name": "TonicIndianMultiPitch",
        "kwargs": {},
    },
    "timbre:mridangam-stroke": {
        "module_name": "compiam.timbre.mridangam_stroke_classification",
        "class_name": "MridangamStrokeClassification",
        "kwargs": {},
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
