import os
import csv
import json
import yaml


def write_csv(data, out_path, header=None):
    """Writing two dimensional data into a file (.csv)

    :param data: the data to write
    :param output_path: the path where the data is going to be stored
    
    :returns: None
    """
    D = list(zip(*data))
    with open(out_path, "w") as f:
        writer = csv.writer(f)
        if header:
            assert len(header) == len(D[0]), "Header and row length mismatch"
            writer.writerow(header)
        for row in D:
            writer.writerow(row)


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
