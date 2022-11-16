import random

from compiam.utils import remove_key


def split_mirdata_tracks(dataloader, split=0.2):
    """Randomly plits dictionary of mirdata tracks into two track dicts given a splitting coef

    :param dataloader: mirdata dataloader
    :param split: coefficient to set the split size
    :returns: two splits of the original track dictionary
    """
    tracks = dataloader.load_tracks()
    ids_to_split = random.sample(dataloader.track_ids, int(len(dataloader.track_ids)*split))
    new_split = {}
    for idx in ids_to_split:
        new_split[idx] = tracks[idx]
        tracks = remove_key(tracks)
    return tracks, new_split