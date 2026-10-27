import numpy as np
import torch
import torch.nn.functional as F


def zeropad(spect, left=0, right=0):
    """Pad a spectrogram of shape (time, bins) with `left` frames at the start and `right` at the end."""
    if left == 0 and right == 0:
        return spect
    return F.pad(spect, (0, 0, left, right), "constant", 0)


def split_spectrogram(spect, chunk_size, border_size=6):
    """Split a spectrogram of shape (time, bins) into overlapping chunks.

    The model was not trained on the edges of its input, hence the predictions of the first and
    last `border_size` frames of every chunk are discarded. To compensate for that, consecutive
    chunks overlap by `border_size` frames, and the first and last chunks are padded. The last
    chunk is shifted to end exactly at the end of the piece, unless the piece is shorter than a chunk.

    :param spect: spectrogram tensor of shape (time, bins).
    :param chunk_size: number of frames of the chunks.
    :param border_size: number of frames discarded at each side of the predictions of a chunk.

    :returns: tuple with the list of chunks and the array of their starting frames.
    """
    starts = np.arange(
        -border_size, len(spect) - border_size, chunk_size - 2 * border_size
    )
    if len(spect) > chunk_size - 2 * border_size:
        starts[-1] = len(spect) - (chunk_size - border_size)
    chunks = [
        zeropad(
            spect[max(start, 0) : min(start + chunk_size, len(spect))],
            left=max(0, -start),
            right=max(0, min(border_size, start + chunk_size - len(spect))),
        )
        for start in starts
    ]
    return chunks, starts


def merge_chunks(predictions, starts, length, chunk_size, border_size=6):
    """Merge the framewise predictions of the chunks of a piece.

    In the frames predicted by several chunks, the earliest chunk takes precedence.

    :param predictions: list of tensors with the predictions of each chunk.
    :param starts: starting frames of the chunks.
    :param length: number of frames of the whole piece.
    :param chunk_size: number of frames of the chunks.
    :param border_size: number of frames discarded at each side of the predictions of a chunk.

    :returns: tensor with the predictions for the whole piece.
    """
    merged = torch.full((length,), -1000.0, device=predictions[0].device)
    # later chunks are written first, so that earlier ones overwrite them
    for start, prediction in zip(reversed(starts), reversed(predictions)):
        if border_size > 0:
            prediction = prediction[border_size:-border_size]
        merged[start + border_size : start + chunk_size - border_size] = prediction
    return merged
