import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils import data
except:
    raise ImportError(
        "In order to use this tool you need to have torch installed. "
        "Install compIAM with torch support: pip install 'compiam[torch]'"
    )


# class for sf layers
class sfmodule(nn.Module):
    def __init__(self, n_ch_in):
        super(sfmodule, self).__init__()
        n_filters = 16
        self.bn1 = nn.BatchNorm2d(n_ch_in, track_running_stats=True)
        self.conv1 = nn.Conv2d(n_ch_in, n_filters, (1, 5), stride=1, padding=(0, 2))
        self.elu = nn.ELU()
        self.do = nn.Dropout(p=0.1)

    def forward(self, x):
        y = self.bn1(x)
        y = self.conv1(x)
        y = self.elu(y)
        y = self.do(y)
        return y


# class for multi-filter module
class mfmodule(nn.Module):
    def __init__(self, pool_height, n_ch, kernel_widths, n_filters):
        super(mfmodule, self).__init__()
        self.avgpool1 = nn.AvgPool2d((pool_height, 1))
        self.bn1 = nn.BatchNorm2d(n_ch, track_running_stats=True)

        self.conv1s = nn.ModuleList([])
        for kw in kernel_widths:
            self.conv1s.append(
                nn.Conv2d(n_ch, n_filters[0], (1, kw), stride=1, padding=(0, kw // 2))
            )

        self.do = nn.Dropout(0.5)
        self.conv2 = nn.Conv2d(
            n_filters[0] * len(kernel_widths), n_filters[1], (1, 1), stride=1
        )

    def forward(self, x):
        y = self.avgpool1(x)
        y = self.bn1(y)
        z = []
        for conv1 in self.conv1s:
            z.append(conv1(y))

        # trim last column to keep width = input_len (needed if filter width is even)
        for i in range(len(z)):
            z[i] = z[i][:, :, :, :-1]

        y = torch.cat(z, dim=1)
        y = self.do(y)
        y = self.conv2(y)
        return y


class densemodule(nn.Module):
    def __init__(self, n_ch_in, input_len, input_height, n_classes):
        super(densemodule, self).__init__()
        n_linear1_in = n_ch_in * input_height

        self.dense_mod = nn.ModuleList(
            [
                nn.AvgPool2d((1, input_len)),
                nn.BatchNorm2d(n_ch_in, track_running_stats=True),
                nn.Dropout(p=0.5),
                nn.Flatten(),
                nn.Linear(n_linear1_in, n_classes),
            ]
        )

    def forward(self, x):
        for layer in self.dense_mod:
            x = layer(x)
        return x


def build_model(input_height, input_len, n_classes):
    model = nn.Sequential()
    i_module = 0

    # add sf layers
    sfmod_ch_sizes = [1, 16, 16]
    for ch in sfmod_ch_sizes:
        sfmod_i = sfmodule(ch)
        model.add_module(str(i_module), sfmod_i)
        i_module += 1

    # add mfmods
    pool_height = 5
    kernel_widths = [16, 32, 64, 96]
    ch_in, ch_out = 16, 16
    mfmod_n_filters = [12, 16]

    mfmod_i = mfmodule(pool_height, ch_in, kernel_widths, mfmod_n_filters)
    model.add_module(str(i_module), mfmod_i)
    input_height //= pool_height
    i_module += 1

    # add densemod
    ch_in = 16
    densemod = densemodule(ch_in, input_len, input_height, n_classes)
    model.add_module(str(i_module), densemod)
    return model


# data-loader(https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)
class Dataset(data.Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(self, datadir, list_IDs, labels):
        "Initialization"
        self.datadir = datadir
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        """Denotes the total number of samples"""
        return len(self.list_IDs)

    def __getitem__(self, index):
        """Generates one sample of data"""
        # Select sample
        ID = self.list_IDs[index]  # os.path.join(self.datadir, self.list_IDs[index])

        # Load data and get label
        # X  =  torch.tensor(np.load(ID))
        X = torch.load(ID)
        y = self.labels[ID.replace(self.datadir, "")]  # .replace(".npy","")]
        id = ID.replace(self.datadir, "")

        return X, y, id


def class_to_categorical(labels, classes):
    map = dict(zip(classes, np.arange(0, len(classes)).tolist()))
    for i in range(len(labels)):
        labels[i] = map[labels[i].item()]
    return labels


def categorical_to_class(class_ids, classes):
    map = dict(zip(np.arange(0, len(classes)).tolist(), classes))
    for i in range(len(class_ids)):
        class_ids[i] = map[class_ids[i].item()]
    return class_ids


# function to smooth predicted s.t.m. estimates by constraining minimum section duration
def smooth_boundaries(stmvstime_track, min_dur):
    stmvstime_track_smu = np.copy(stmvstime_track)
    prev_stm = stmvstime_track_smu[0]
    curr_stm_dur = 1
    i = 1
    while i < len(stmvstime_track_smu):
        if stmvstime_track_smu[i] != stmvstime_track_smu[i - 1]:
            if curr_stm_dur >= min_dur:
                curr_stm_dur = 1
                prev_stm = stmvstime_track_smu[i - 1]

            else:
                # if stmvstime_track_smu[i] =  = prev_stm:
                stmvstime_track_smu[i - curr_stm_dur : i] = prev_stm
                # else:
                # 	prev_stm = stmvstime_track_smu[i-1]
                curr_stm_dur = 1
        else:
            curr_stm_dur += 1
        i += 1
    return stmvstime_track_smu
