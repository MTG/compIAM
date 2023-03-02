import os
import csv
import json
import yaml
import pickle

import numpy as np

from compiam import utils


def write_csv(data, out_path, header=None):
    """Writing multi-dimensional data into a file (.csv)

    :param data: the data to write
    :param output_path: the path where the data is going to be stored

    :returns: None
    """
    data = np.array(data)
    with open(out_path, "w") as f:
        writer = csv.writer(f, delimiter=",")
        if header:
            if len(header) != len(data[0, :]):
                raise ValueError("Header and row length mismatch")
            writer.writerow(header)
        writer.writerows(data)


def read_csv(file_path):
    """Reading a csv file (.csv)

    :param file_path: path to the csv

    :returns: numpy array containing the data from the read CSV
    """
    output = np.genfromtxt(file_path, delimiter=",")
    return output[~np.isnan(output)]


def save_object(obj, filename):
    with open(filename, "wb") as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def write_json(j, path):
    """
    Write json, <j>, to <path>

    :param j: json
    :type path: json
    :param path: path to write to,
        if the directory doesn't exist, one will be created
    :type path: str
    """
    utils.create_if_not_exists(path)
    # Opening JSON file
    with open(path, "w") as f:
        json.dump(j, f)


#####################
# Dunya writing utils
#####################
def write_json(sections, output_path):
    """Writing json-based data into a file (.json)

    :param data: the data to write
    :param output_path: the path where the data is going to be stored

    :returns: None
    """
    output_path = output_path + ".json"
    with open(output_path, "w") as fhandle:
        json.dump(sections, fhandle)
    fhandle.close()


def write_scalar_txt(data, output_path):
    """Writing scalar data into a file (.txt)

    :param data: the data to write
    :param output_path: the path where the data is going to be stored

    :returns: None
    """
    output_path = output_path + ".txt"
    with open(output_path, "w") as f:
        f.write("{}".format(data))
    f.close()


def load_yaml(path):
    """Load yaml at <path> to dictionary, d

    :param path: input file
    """
    import zope.dottedname.resolve

    def constructor_dottedname(loader, node):
        value = loader.construct_scalar(node)
        return zope.dottedname.resolve.resolve(value)

    yaml.add_constructor("!dottedname", constructor_dottedname)

    if not os.path.isfile(path):
        return None
    with open(path) as f:
        d = yaml.load(f, Loader=yaml.FullLoader)
    return d
