from compiam.rhythm.tabla_transcription.models import onsetCNN, onsetCNN_D, onsetCNN_RT

#TODO: think about global variables
paths_dict = {
	'1-way_B':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/B/saved_model_0.pt',
		'model': onsetCNN()
		},
	'1-way_B':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/B/saved_model_1.pt',
		'model': onsetCNN()
		},
	'1-way_B':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/B/saved_model_2.pt',
		'model': onsetCNN()
		},
	'1-way_D':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/D/saved_model_0.pt',
		'model': onsetCNN_D()
		},
	'1-way_D':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/D/saved_model_1.pt',
		'model': onsetCNN_D()
		},
	'1-way_D':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/D/saved_model_2.pt',
		'model': onsetCNN_D()
		},
	'1-way_RB':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/RB/saved_model_0.pt',
		'model': onsetCNN()
		},
	'1-way_RB':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/RB/saved_model_1.pt',
		'model': onsetCNN()
		},
	'1-way_RB':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/RB/saved_model_2.pt',
		'model': onsetCNN()
		},
	'1-way_RT':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/RT/saved_model_0.pt',
		'model': onsetCNN_RT()
		},
	'1-way_RT':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/RT/saved_model_1.pt',
		'model': onsetCNN_RT()
		},
	'1-way_RT':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/1-way/RT/saved_model_2.pt',
		'model': onsetCNN_RT()
		},
	'B':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/B/saved_model_0.pt',
		'model': onsetCNN()
		},
	'B':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/B/saved_model_1.pt',
		'model': onsetCNN()
		},
	'B':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/B/saved_model_2.pt',
		'model': onsetCNN()
		},
	'D':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/D/saved_model_0.pt',
		'model': onsetCNN_D()
		},
	'D':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/D/saved_model_1.pt',
		'model': onsetCNN_D()
		},
	'D':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/D/saved_model_2.pt',
		'model': onsetCNN_D()
		},
	'RB':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/RB/saved_model_0.pt',
		'model': onsetCNN()
		},
	'RB':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/RB/saved_model_1.pt',
		'model': onsetCNN()
		},
	'RB':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/RB/saved_model_2.pt',
		'model': onsetCNN()
		},
	'RT':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/RT/saved_model_0.pt',
		'model': onsetCNN_RT()
		},
	'RT':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/RT/saved_model_1.pt',
		'model': onsetCNN_RT()
		},
	'RT':
		{
		'path':'../4way-tabla-transcription/code/stroke-classification/saved_models/RT/saved_model_2.pt'
		'model': onsetCNN_RT()
		}
}