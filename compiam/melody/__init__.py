from compiam.melody import FTANetCarnatic

class ftanetCarnatic:
    def __init__(self, filepath):
        self.filepath = filepath
        self.model = FTANetCarnatic
        self.model.load_weights(filepath)

    def predict(self, path_to_audio, sample_rate=44100, hop_size=80, batch_size=5):
        pitch_track = self.model.extract(path_to_audio, sample_rate=sample_rate, hop_size=hop_size, batch_size=batch_size)
        return pitch_track