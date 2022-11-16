"""
Created on April 13, 2018
Edited on July 05, 2019

@author: Gaetan Hadjeres & Stefan Lattner

Sony CSL Paris, France
"""
import os
import bz2

import soundfile
import torch
import pickle
import logging

import numpy as np

from pickle import UnpicklingError
from torch.autograd import Variable

LOGGER = logging.getLogger(__name__)


def normalize(x):
    x -= np.min(x)
    x /= np.max(x) + 1e-8
    return x


def read_file(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            lines[i] = line.strip()

    return lines


def check_audio_files(filelist):
    for file in filelist:
        assert os.path.exists(file), f'File does not exist: {file}'
        assert os.path.isfile(file), f'Not a file: {file}'


def prepare_audio_inputs(input_files):
    input_files = read_file(input_files)
    check_audio_files(input_files)
    return input_files

def cuda_tensor(data, tocuda=True):
    if torch.cuda.is_available() and tocuda:
        return torch.FloatTensor(data).cuda()
    else:
        return torch.FloatTensor(data)


def cuda_variable(tensor, volatile=False):
    if torch.cuda.is_available():
        try:
            return Variable(tensor.cuda(), volatile=volatile)
        except AttributeError:
            return Variable(torch.Tensor(tensor).cuda(), volatile=volatile)
    else:
        try:
            return Variable(tensor, volatile=volatile)
        except TypeError:
            return Variable(torch.Tensor(tensor), volatile=volatile)


def to_numpy(variable):
    if type(variable) == np.ndarray:
        return variable
    try:
        if torch.cuda.is_available():
            return variable.data.cpu().numpy()
        else:
            return variable.data.numpy()
    except:
        try:
            return variable.numpy()     
        except:
            LOGGER.warning("Could not 'to_numpy' variable of type "
                           f"{type(variable)}")
            return variable


def save_pyc_bz(data, fn):
    """
    Saves data to file (bz2 compressed)

    :param data: data to save
    :param fn: file name of dumped data
    """
    pickle.dump(data, bz2.BZ2File(fn, 'w'))


def load_pyc_bz(fn):
    """
    Loads data from file (bz2 compressed)

    :param fn: file name of dumped data
    :return: loaded data
    """
    try:
        return pickle.load(bz2.BZ2File(fn, 'r'), encoding='latin1')
    except EOFError:
        return pickle.load(bz2.BZ2File(fn, 'r'))


def cached(cache_fn, func, args=(), kwargs={}, refresh_cache=False,
           logger=None):
    """
    If `cache_fn` exists, return the unpickled contents of that file
    (the cache file is treated as a bzipped pickle file). If this
    fails, compute `func`(*`args`), pickle the result to `cache_fn`,
    and return the result.

    Parameters
    ----------

    func : function
        function to compute

    args : tuple
        argument for which to evaluate `func`

    cache_fn : str
        file name to load the computed value `func`(*`args`) from

    refresh_cache : boolean
        if True, ignore the cache file, compute function, and store the result
        in the cache file

    Returns
    -------

    object

        the result of `func`(*`args`)

    """
    if logger==None:
        LOGGER = logging.getLogger(__name__)
    else:
        LOGGER = logger

    result = None
    if cache_fn is not None and os.path.exists(cache_fn):
        if refresh_cache:
            os.remove(cache_fn)
        else:
            try:
                LOGGER.info(f"Loading cache file {cache_fn}...")
                result = load_pyc_bz(cache_fn)
            except UnpicklingError as e:
                LOGGER.error(
                    ('The file {0} exists, but cannot be unpickled.'
                     'Is it readable? Is this a pickle file? Try '
                     'with numpy..'
                     '').format(cache_fn))
                try:
                    result = np.load(cache_fn)
                except Exception as g:
                    LOGGER.error("Did not work, either.")
                    raise e

    if result is None:
        result = func(*args, **kwargs)
        if cache_fn is not None:
            try:
                save_pyc_bz(result, cache_fn)
            except Exception as e:
                LOGGER.error("Could not save, try with numpy..")
                try:
                    np.save(cache_fn, result)
                except Exception as g:
                    LOGGER.error("Did not work, either.")
                    raise e
    return result
