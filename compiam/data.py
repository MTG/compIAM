#############
# Models Dict
#############
# Each model should be stored in models_dict using <name>:<d> where <d> is:
#	{'filepath': '<path_to_model>', 'wrapper': model wrapper}
from compiam.melody import ftanetCarnatic, Melodia, TonicIndianMultiPitch
from compiam.rhythm import fourWayTabla
from compiam.timbre import MridangamStrokeClassification

models_dict = {
    'rhythm:1way-tabla': {
        'wrapper': fourWayTabla,
        'kwargs': {'filepath': 'models/rhythm/4wayTabla/1way/'}
    },
    'rhythm:4way-tabla': {
        'wrapper': fourWayTabla,
        'kwargs': {'filepath': 'models/rhythm/4wayTabla/4way/'}
    },
    'melody:ftanet-carnatic': {
        'wrapper': ftanetCarnatic,
        'kwargs': {'filepath': 'models/melody/ftanet/carnatic/'}
    },
    'melody:melodia': {
        'wrapper': Melodia,
        'kwargs': {}
    },
    'melody:tonic-multipitch': {
        'wrapper': TonicIndianMultiPitch,
        'kwargs': {}
    },
    'timbre:mridangam-stroke': {
        'wrapper': MridangamStrokeClassification,
        'kwargs': {}
    },
}

###############
# Datasets List
###############
datasets_list = ['saraga_carnatic', 'saraga_hindustani', 'mridangam_stroke']
