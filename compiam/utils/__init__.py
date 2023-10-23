import os
import logging
import inspect
import pathlib
import pickle
import difflib
import librosa

import IPython.display as ipd
import numpy as np

from compiam.io import save_object, load_yaml
from compiam.utils.pitch import cents_to_pitch

WORKDIR = os.path.dirname(pathlib.Path(__file__).parent.resolve())

svara_cents_carnatic_path = os.path.join(WORKDIR, "conf", "raga", "svara_cents.yaml")
svara_lookup_carnatic_path = os.path.join(WORKDIR, "conf", "raga", "carnatic.yaml")


def get_logger(name):
    """Create logger

    :param name: logger name
    """
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    return logger


logger = get_logger(__name__)


def create_if_not_exists(path):
    """If the directory at <path> does not exist, create it empty

    :param path: path to folder
    """
    directory = os.path.dirname(path)
    # Do not try and create directory if path is just a filename
    if (not os.path.exists(directory)) and (directory != ""):
        os.makedirs(directory)


def get_tool_list(modules):
    """Given sys.modules[__name__], prints out the imported classes

    :param modules: basically the sys.modules[__name__] of a file
    """
    list_of_tools = []
    for _, obj in inspect.getmembers(modules):
        if inspect.isclass(obj):
            list_of_tools.append(obj.__name__)
    return list_of_tools


def run_or_cache(func, inputs, cache):
    """
    Run function, <func> with inputs, <inputs> and save
    to <cache>. If <cache> already exists, load rather than
    run anew

    :param func: python function
    :type func: function
    :param inputs: parameters to pass to <func>, in order
    :type inputs: tuple
    :param cache: .pkl filepath
    :type cache: str or None

    :returns: output of <func>
    :rtype: equal to type returned by <func>
    """
    if cache:
        if os.path.isfile(cache):
            try:
                file = open(cache, "rb")
                results = pickle.load(file)
                return results
            except:
                logger.warning("Error loading from cache, recomputing")
    results = func(*inputs)

    if cache:
        try:
            create_if_not_exists(cache)
            save_object(results, cache)
        except Exception as e:
            logger.error(f"Error saving object: {e}")

    return results


def myround(x, base=5):
    return base * round(x / base)


def get_timestamp(secs, divider="-"):
    """
    Convert seconds into timestamp

    :param secs: seconds
    :type secs: int
    :param divider: divider between minute and second, default "-"
    :type divider: str

    :return: timestamp
    :rtype: str
    """
    minutes = int(secs / 60)
    seconds = round(secs % 60, 2)
    return f"{minutes}min{divider}{seconds}sec"


def ipy_audio(y, t1, t2, sr=44100):
    y_ = y[round(t1 * sr) : round(t2 * sr)]
    return ipd.Audio(y_, rate=sr, autoplay=False)


def get_svara_pitch_carnatic(raga, tonic=None):
    svara_pitch = get_svara_pitch(
        raga, tonic, svara_cents_carnatic_path, svara_lookup_carnatic_path
    )
    return svara_pitch


def get_svara_pitch(raga, tonic, svara_cents_path, svara_lookup_path):
    svara_cents = load_yaml(svara_cents_path)
    svara_lookup = load_yaml(svara_lookup_path)

    if not raga in svara_lookup:
        all_ragas = list(svara_lookup.keys())
        close = difflib.get_close_matches(raga, all_ragas)
        error_message = f"Raga, {raga} not available in conf."
        if close:
            error_message += f" Nearest matches: {close}"
        raise ValueError(error_message)

    arohana = svara_lookup[raga]["arohana"]
    avorohana = svara_lookup[raga]["avorohana"]
    all_svaras = list(set(arohana + avorohana))

    if tonic:
        svara_pitch = {cents_to_pitch(k, tonic): v for k, v in svara_cents.items()}
    else:
        svara_pitch = svara_cents

    final_dict = {}
    for svara in all_svaras:
        for c, sl in svara_pitch.items():
            for s in sl:
                if svara == s:
                    final_dict[c] = s

    return final_dict


def add_center_to_mask(mask):
    num_one = 0
    indices = []
    for i, s in enumerate(mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            li = len(indices)
            if li:
                middle = indices[int(li / 2)]
                mask[middle] = 2
                num_one = 0
                indices = []
    return mask
