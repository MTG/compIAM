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
#           "default_version": "<version code of default version>",
#           "kwargs": {
#               "<version code>": {
#                   "model_path": "<path_to_model>",
#                   "download_link": "<link_to_download_model>",
#                   "download_checksum": "<checksum_of_downloaded_model>",
#                   other arguments...
#               },
#               "<version code 2 (if applicable)>": {
#                   arguments...
#               },
#               more versions...
#           },
#       }

models_dict = {
    "melody:deepsrgm": {
        "module_name": "compiam.melody.raga_recognition.deepsrgm",
        "class_name": "DEEPSRGM",
        "default_version": "v1",
        "kwargs": {
            "v1": {
                "model_path": {
                    "lstm": os.path.join(
                        "models",
                        "melody",
                        "deepsrgm",
                        "baseline",
                        "lstm_25_checkpoint.pth",
                    ),
                    "gru": os.path.join(
                        "models",
                        "melody",
                        "deepsrgm",
                        "baseline",
                        "gru_30_checkpoint.pth",
                    ),
                },
                "mapping_path": os.path.join(
                    "models",
                    "melody",
                    "deepsrgm",
                    "baseline",
                    "raga_mapping.json",
                ),
                "download_link": "https://zenodo.org/records/13984096/files/deepsrgm.zip?download=1",
                "download_checksum": "dc7560af17e546f161786a8cbd47727e",
            },
        },
    },
    "melody:ftanet-carnatic": {
        "module_name": "compiam.melody.pitch_extraction.ftanet_carnatic",
        "class_name": "FTANetCarnatic",
        "default_version": "v1",
        "kwargs": {
            "v1": {
                "model_path": os.path.join(
                    "models", "melody", "ftanet-carnatic", "carnatic", "carnatic"
                ),
                "download_link": "https://zenodo.org/records/13981708/files/FTANetCarnatic-vocals.zip?download=1",
                "download_checksum": "e8bb3a86eef6beae11bf1eb49a57cbbf",
                "sample_rate": 8000,
            }
        },
    },
    "melody:ftaresnet-carnatic-violin": {
        "module_name": "compiam.melody.pitch_extraction.ftaresnet_carnatic",
        "class_name": "FTAResNetCarnatic",
        "default_version": "v1",
        "kwargs": {
            "v1": {
                "model_path": os.path.join(
                    "models",
                    "melody",
                    "ftaresnet-carnatic",
                    "carnatic_violin",
                    "FTA-ResNet_best_version.pth",
                ),
                "sample_rate": 44100,
                "download_link": "https://zenodo.org/records/13983762/files/FTAResNetCarnatic-violin.zip?download=1",
                "download_checksum": "bda6e841b0e8ec7418bf9e0e9eb57a19",
            }
        },
    },
    "melody:cae-carnatic": {
        "module_name": "compiam.melody.pattern.sancara_search",
        "class_name": "CAEWrapper",
        "default_version": "v1",
        "kwargs": {
            "v1": {
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
                "download_link": "https://zenodo.org/records/13984138/files/caecarnatic.zip?download=1",
                "download_checksum": "81962c8866bb6e76023d6e1dc5d1904d",
            },
        },
    },
    "structure:dhrupad-bandish-segmentation": {
        "module_name": "compiam.structure.segmentation.dhrupad_bandish_segmentation",
        "class_name": "DhrupadBandishSegmentation",
        "default_version": "v1",
        "kwargs": {
            "v1": {
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
                "splits_path": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "splits",
                ),
                "annotations_path": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "annotations",
                ),
                "features_path": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "features",
                ),
                "original_audios_path": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "audio_original",
                ),
                "processed_audios_path": os.path.join(
                    "models",
                    "structure",
                    "dhrupad_bandish_segmentation",
                    "baseline",
                    "audio_sections",
                ),
                "download_link": "https://zenodo.org/records/13984151/files/dhrupad_bandish_segmentation.zip?download=1",
                "download_checksum": "37967af81b0020a516b2470f659b481c",
            },
        },
    },
    "separation:cold-diff-sep": {
        "module_name": "compiam.separation.singing_voice_extraction.cold_diff_sep",
        "class_name": "ColdDiffSep",
        "default_version": "v1",
        "kwargs": {
            "v1": {
                "model_path": os.path.join(
                    "models",
                    "separation",
                    "cold_diff_sep",
                    "saraga-8",
                    "saraga-8.ckpt-1",
                ),
                "download_link": "https://zenodo.org/records/13984075/files/cold-diff-sep.zip?download=1",
                "download_checksum": "cdd98fa8725ec1efd4017c34e2ff8ce6",
            },
            # "sample_rate": 22050,  # Already contained in the model config
        },
    },
    "separation:mixer-model": {
        "module_name": "compiam.separation.music_source_separation.mixer_model",
        "class_name": "MixerModel",
        "default_version": "v1",
        "kwargs": {
            "v1": {
                "model_path": os.path.join(
                    "models",
                    "separation",
                    "mixer_model",
                    "vocal_violin",
                    "mdx_model_vocal_violin_600k.pth",
                ),
                "download_link": "https://zenodo.org/records/14082164/files/mixer_model.zip?download=1",
                "download_checksum": "4d48b6ad248d20c3ed9891a7e392621d",
            },
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
