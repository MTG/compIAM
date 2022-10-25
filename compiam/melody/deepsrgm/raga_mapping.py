import os
import json

from compiam.data import WORKDIR


def create_mapping(mapping_path, selection):
    """Creating a map for the ragas available to us in the dataset (40 out of 71)
    """
    with open(mapping_path, "r") as fhandle:
        legend = json.load(fhandle)

    # Create mapping with raga and iids
    keys = list(legend.keys())
    mapping = dict()
    for key in keys:
        if key in legend.keys():
            mapping[key] = legend[key]

    # integer and simple ID per raga
    index2hash = dict()
    for i, cls in enumerate(mapping.keys()):
        index2hash[i] = cls

    # Select determined raagas
    final_map = dict()
    for i, cls in enumerate(selection):
        final_map[i] = mapping[index2hash[cls]]
    
    return final_map