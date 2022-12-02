import csv
import json
import numpy as np
import pickle

import compiam.utils

def write_csv(data, out_path, header=None):
    D = list(zip(*data))
    with open(out_path, "w") as f:
        writer = csv.writer(f)
        if header:
            assert len(header) == len(D[0]), "Header and row length mismatch"
            writer.writerow(header)
        for row in D:
            writer.writerow(row)


def save_object(obj, filename):
    import pickle
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
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
    compiam.utils.create_if_not_exists(path)
    # Opening JSON file 
    with open(path, 'w') as f:
        json.dump(j, f)


#####################
# Dunya writing utils
#####################
def write_2d_csv(data, output_path):
    """Writing two dimensional data into a file (.csv)
    :param data: the data to write
    :param output_path: the path where the data is going to be stored
    :returns: None
    """
    output_path = output_path + ".csv"
    data = np.array(data)
    with open(output_path, "w") as f:
        for i, j in zip(data[:, 0], data[:, 1]):
            f.write("{}, {}\n".format(i, j))
    f.close()


def write_1d_csv(data, output_path):
    """Writing one dimensional data into a file (.csv)
    :param data: the data to write
    :param output_path: the path where the data is going to be stored
    :returns: None
    """
    output_path = output_path + ".csv"
    with open(output_path, "w") as f:
        for i in data:
            f.write("{}\n".format(i))
    f.close()


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
