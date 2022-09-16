#############
# Models Dict
#############
# Each model should be stored in models_dict using <name>:<d> where <d> is:
#	{'filepath': '<path_to_model>', 'wrapper': model wrapper}
from compiam.melody import ftanetCarnatic
from compiam.rhythm import fourWayTabla

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
	}
}

###############
# Datasets List
###############
datasets_list = ['saraga_carnatic', 'saraga_hindustani', 'mridangam_stroke']