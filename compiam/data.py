#############
# Models Dict
#############
# Each model should be stored in models_dict using <name>:<d> where <d> is:
#	{"filepath": "<path_to_model>", "wrapper": model wrapper}
from compiam.melody import ftanetCarnatic, Melodia, TonicIndianMultiPitch
from compiam.rhythm import fourWayTabla
from compiam.timbre import MridangamStrokeClassification

models_dict = {
    "rhythm:1way-tabla": {
        "wrapper": fourWayTabla,
        "kwargs": {"filepath": "models/rhythm/4wayTabla/1way/"}
    },
    "rhythm:4way-tabla": {
        "wrapper": fourWayTabla,
        "kwargs": {"filepath": "models/rhythm/4wayTabla/4way/"}
    },
    "melody:ftanet-carnatic": {
        "wrapper": ftanetCarnatic,
        "kwargs": {"filepath": "models/melody/ftanet/carnatic/"}
    },
    "melody:melodia": {
        "wrapper": Melodia,
        "kwargs": {}
    },
    "melody:tonic-multipitch": {
        "wrapper": TonicIndianMultiPitch,
        "kwargs": {}
    },
    "timbre:mridangam-stroke": {
        "wrapper": MridangamStrokeClassification,
        "kwargs": {}
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

