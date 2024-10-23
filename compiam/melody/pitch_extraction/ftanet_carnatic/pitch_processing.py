# The functions in this file were directly ported (and in some cases adapted)
# from the original repoository ofthe FTA-Net (https://github.com/yushuai/FTANet-melodic).
# We documented these for a more clear usage of the code by the CompIAM users. For direct
# use of the FTA-Net as designed and published by the authors, please refer to the original
# mentioned GitHub repository.

import numpy as np

from compiam.melody.pitch_extraction.ftanet_carnatic.cfp import get_CenFreq


def batchize_test(data, size=430):
    """Re-arrange CFP features to fit the FTA-Net model.

    :param data: input CFP features for the FTA-Net.
    :param size: size of the batches.
    :returns: batched features.
    """
    xlist = []
    num = int(data.shape[-1] / size)
    if data.shape[-1] % size != 0:
        num += 1
    for i in range(num):
        if (i + 1) * size > data.shape[-1]:
            batch_x = np.zeros((data.shape[0], data.shape[1], size))

            tmp_x = data[:, :, i * size :]

            batch_x[:, :, : tmp_x.shape[-1]] += tmp_x
            xlist.append(batch_x.transpose(1, 2, 0))
            break
        else:
            batch_x = data[:, :, i * size : (i + 1) * size]
            xlist.append(batch_x.transpose(1, 2, 0))

    return np.array(xlist)


def est(output, CenFreq, time_arr):
    """Re-arrange FTA-Net output to a versatile pitch time-series.

    :param data: input CFP features for the FTA-Net.
    :param size: size of the batches.
    :returns: batched features.
    """
    CenFreq[0] = 0
    est_time = time_arr
    est_freq = np.argmax(output, axis=0)

    for j in range(len(est_freq)):
        est_freq[j] = CenFreq[int(est_freq[j])]

    if len(est_freq) != len(est_time):
        new_length = min(len(est_freq), len(est_time))
        est_freq = est_freq[:new_length]
        est_time = est_time[:new_length]

    est_arr = np.concatenate((est_time[:, None], est_freq[:, None]), axis=1)

    return est_arr


def iseg(data):
    """Re-shape data.

    :param data: input features.
    :returns: re-shaped data.
    """
    # data: (batch_size, freq_bins, seg_len)
    new_length = data.shape[0] * data.shape[-1]  # T = batch_size * seg_len
    new_data = np.zeros((data.shape[1], new_length))  # (freq_bins, T)
    for i in range(len(data)):
        new_data[:, i * data.shape[-1] : (i + 1) * data.shape[-1]] = data[i]
    return new_data


def get_est_arr(model, x_list, y_list, batch_size):
    """Run the FTA-Net model in batches and construct the final pitch time-series.

    :param model: built and trained model.
    :param x_list: features.
    :param y_list: timestamps.
    :param batch_size: batch size of the input data.
    :returns: output pitch time-series.
    """
    for i in range(len(x_list)):
        x = x_list[i]
        y = y_list[i]

        # predict and concat
        num = x.shape[0] // batch_size
        if x.shape[0] % batch_size != 0:
            num += 1
        preds = []
        for j in range(num):
            # x: (batch_size, freq_bins, seg_len)
            if j == num - 1:
                X = x[j * batch_size :]
                length = x.shape[0] - j * batch_size
            else:
                X = x[j * batch_size : (j + 1) * batch_size]
                length = batch_size

            prediction = model.predict(X, length)
            preds.append(prediction)

        # (num*bs, freq_bins, seg_len) to (freq_bins, T)
        preds = np.concatenate(preds, axis=0)
        preds = iseg(preds)
        # transform to f0ref
        CenFreq = get_CenFreq(StartFreq=31, StopFreq=1250, NumPerOct=60)
        est_arr = est(preds, CenFreq, y)

    return est_arr


def std_normalize(data):
    """Standardize the input data.

    :param data: input data.
    :returns: standardized data.
    """
    data = data.astype(np.float64)
    mean = np.mean(data)
    std = np.std(data)
    data = data.copy() - mean
    if std != 0.0:
        data = data / std
    return data.astype(np.float32)
