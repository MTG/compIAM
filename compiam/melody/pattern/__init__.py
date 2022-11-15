import numpy as np

class SancaraSearcher:
	def __init__(self):
		self.pitch, self.time, self.timestep = self.pitch_track_or_file(pitch_track)
		self.stability_track = self.
		self.features = 
		self.self_sim = 

	def pitch_track_or_file(self, pt):
		if isinstance(pt, str):
			pitch, time, timestep = self._get_timeseries(pt)
		elif isinstance(pt, [np.array, list]):
			if isinstance(pt, list):
				pt = np.array(list)
			if pt.shape[1] != 2:
				raise ValueError('Pitch track should be of the form [(time, pitch value),...] or path to pitch track in this format.')
			time = pt[:,0]
			pitch = pt[:,1]
			timestep = time[3] - time[2]
		else:
			raise TypeError('<pitch_track> should be filepath to pitch track or precomputed iterable corresponding to pitch track.')
		return pitch, time, timestep

	def _get_timeseries(self, path):
	    pitch = []
	    time = []
	    with open(path, 'r') as f:
	        for i in f:
	            t, f = i.replace('/n','').split(',')
	            pitch.append(float(f))
	            time.append(float(t))
	    timestep = time[3]-time[2]
	    return np.array(pitch), np.array(time), timestep