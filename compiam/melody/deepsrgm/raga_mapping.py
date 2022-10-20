import os
import json

from compiam.data import WORKDIR

def load_legend():
    with open(os.path.join(WORKDIR, "conf", "raga", "DEEPSRGM_mapping.json"), "r") as fhandle:
        legend = json.load(fhandle)
    return legend

# creating a map for the ragas available to us in the dataset (40 out of 71)
def create_mapping(legend):
    keys = os.listdir()
    mapping = dict()
    for key in keys:
        if key in legend.keys():
            mapping[key] = legend[key]





index2hash=dict()
for i, cls in enumerate(mapping.keys()):
    index2hash[i]=cls
    print(i, mapping[cls])



# show classses and class labels for 10 rag dataset (arbitrarily selected)
selection = [5, 8, 10, 13, 17, 20, 22, 23, 24, 28]
map1 = dict()
for i, cls in enumerate(selection):
    print(i, mapping[index2hash[cls]])
    map1[i] = mapping[index2hash[cls]]