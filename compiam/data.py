import os
import pathlib

WORKDIR = os.path.join(
    os.path.dirname(pathlib.Path(__file__).parent.resolve()), "compiam"
)
TESTDIR = os.path.join(
    os.path.dirname(pathlib.Path(__file__).parent.resolve()), "tests"
)


#############
# Models Dict
#############
# Each model should be stored in models_dict using
#   "<melody|rhythm|timbrestructure>:model_id":<d> where <d> is:
# 	    {
#           "module_name": "<compiam.melody|rhythm|timbre|structure.task.file|folder name of model>",
#           "class_name": "<name of the model class>",
#           "model_path": "<path_to_model>"
#       }

models_dict = {
    "melody:deepsrgm": {
        "module_name": "compiam.melody.raga_recognition.deepsrgm",
        "class_name": "DEEPSRGM",
        "kwargs": {
            "model_path": {
                "lstm": os.path.join(
                    "models", "melody", "deepsrgm", "baseline", "lstm_25_checkpoint.pth"
                ),
                "gru": os.path.join(
                    "models", "melody", "deepsrgm", "baseline", "gru_30_checkpoint.pth"
                ),
            },
            "mapping_path": os.path.join(
                "models",
                "melody",
                "deepsrgm",
                "baseline",
                "raga_mapping.json",
            ),
        },
    },
    "melody:ftanet-carnatic": {
        "module_name": "compiam.melody.pitch_extraction.ftanet_carnatic",
        "class_name": "FTANetCarnatic",
        "kwargs": {
            "model_path": os.path.join(
                "models", "melody", "ftanet", "carnatic", "carnatic"
            ),
            "sample_rate": 8000,
        },
    },
    "melody:cae-carnatic": {
        "module_name": "compiam.melody.pattern.sancara_search",
        "class_name": "CAEWrapper",
        "kwargs": {
            "model_path": os.path.join(
                "models",
                "melody",
                "caecarnatic",
                "carnatic",
                "model_complex_auto_cqt.save",
            ),
            "conf_path": os.path.join(
                "models", "melody", "caecarnatic", "carnatic", "config_cqt.ini"
            ),
            "spec_path": os.path.join(
                "models", "melody", "caecarnatic", "carnatic", "config_spec.cfg"
            ),
        },
    },
    "structure:dhrupad-bandish-segmentation": {
        "module_name": "compiam.structure.segmentation.dhrupad_bandish_segmentation",
        "class_name": "DhrupadBandishSegmentation",
        "kwargs": {
            "model_path": {
                "net": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "pretrained_models",
                    "net",
                ),
                "pakh": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "pretrained_models",
                    "pakh",
                ),
                "voc": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "pretrained_models",
                    "voc",
                ),
            },
        },
    },
    "separation:cold-diff-sep": {
        "module_name": "compiam.separation.singing_voice_extraction.cold_diff_sep",
        "class_name": "ColdDiffSep",
        "kwargs": {
            "model_path": os.path.join(
                "models",
                "separation",
                "cold_diff_sep",
                "saraga-8",
                "saraga-8.ckpt-1",
            ),
            "sample_rate": 22050,
        },
    },
}


###############
# Datasets List
###############
# Make sure you:
#   1. create a dataset loader in mirdata (https://github.com/mir-dataset-loaders/mirdata)
#   2. add the dataset identifier in this list

datasets_list = [
    "saraga_carnatic",
    "saraga_hindustani",
    "mridangam_stroke",
    "four_way_tabla",
    "compmusic_carnatic_rhythm",
    "compmusic_hindustani_rhythm",
    "compmusic_raga",
    "compmusic_indian_tonic",
    "compmusic_carnatic_varnam",
    "scms",
]


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
