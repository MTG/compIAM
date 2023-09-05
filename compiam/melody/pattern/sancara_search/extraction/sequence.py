import numpy as np


def contains_silence(seq, thresh=0.15):
    """If more than <thresh> of <seq> is 0, return True"""
    return sum(seq == 0) / len(seq) > thresh


def moving_average(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def too_stable(seq, dev_thresh=5, perc_thresh=0.63, window=200):
    """If a sufficient proportion of <seq> is "stable" return True"""
    if window > len(seq):
        window = len(seq)
    mu_ = seq[: window - 1]
    mu = np.concatenate([mu_, moving_average(seq, window)])

    dev_arr = abs(mu - seq)
    dev_seq = dev_arr[np.where(~np.isnan(dev_arr))]
    bel_thresh = dev_seq < dev_thresh

    perc = np.count_nonzero(bel_thresh) / len(dev_seq)

    if perc >= perc_thresh:
        is_stable = 1
    else:
        is_stable = 0

    return is_stable


def start_with_silence(seq):
    return any([seq[0] == 0, all(seq[:100] == 0)])


def min_gap(seq, length=86):
    seq2 = np.trim_zeros(seq)
    m1 = np.r_[False, seq2 == 0, False]
    idx = np.flatnonzero(m1[:-1] != m1[1:])
    if len(idx) > 0:
        out = idx[1::2] - idx[::2]
        if any(o >= length for o in out):
            return True
    return False


def is_stable(seq, max_var):
    if None in seq:
        return 0
    seq = seq.astype(float)
    mu = np.nanmean(seq)

    maximum = np.nanmax(seq)
    minimum = np.nanmin(seq)
    if (maximum < mu + max_var) and (minimum > mu - max_var):
        return 1
    else:
        return 0


def reduce_stability_mask(stable_mask, min_stability_length_secs, timestep):
    min_stability_length = int(min_stability_length_secs / timestep)
    num_one = 0
    indices = []
    for i, s in enumerate(stable_mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            if num_one < min_stability_length:
                for ix in indices:
                    stable_mask[ix] = 0
            num_one = 0
            indices = []
    return stable_mask


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


def add_border_to_mask(mask):
    num_one = 0
    indices = []
    for i, s in enumerate(mask):
        if s == 1:
            num_one += 1
            indices.append(i)
        else:
            li = len(indices)
            if li:
                end = indices[-1]
                start = indices[0]
                mask[end] = 2
                mask[start] = 2
                num_one = 0
                indices = []
    return mask


def get_stability_mask(
    raw_pitch, min_stability_length_secs, stability_hop_secs, var_thresh, timestep
):
    stab_hop = int(stability_hop_secs / timestep)
    reverse_raw_pitch = np.flip(raw_pitch)

    # apply in both directions to array to account for hop_size errors
    stable_mask_1 = [
        is_stable(raw_pitch[s : s + stab_hop], var_thresh)
        for s in range(len(raw_pitch))
    ]
    stable_mask_2 = [
        is_stable(reverse_raw_pitch[s : s + stab_hop], var_thresh)
        for s in range(len(reverse_raw_pitch))
    ]

    silence_mask = raw_pitch == 0

    zipped = zip(stable_mask_1, np.flip(stable_mask_2), silence_mask)

    stable_mask = np.array([int((any([s1, s2]) and not sil)) for s1, s2, sil in zipped])

    stable_mask = reduce_stability_mask(
        stable_mask, min_stability_length_secs, timestep
    )

    stable_mask = add_center_to_mask(stable_mask)

    return stable_mask


def convert_seqs_to_timestep(all_groups, cqt_window, sr, timestep):
    lengths = []
    starts = []
    for group in all_groups:
        this_l = []
        this_s = []
        for x0, x1 in group:
            l = x1 - x0
            s = x0
            if l > 0:
                this_l.append(l)
                this_s.append(s)
        lengths.append(this_l)
        starts.append(this_s)

    starts_sec = [[x * cqt_window / sr for x in p] for p in starts]
    lengths_sec = [[x * cqt_window / sr for x in l] for l in lengths]

    starts_seq = [[int(x / timestep) for x in p] for p in starts_sec]
    lengths_seq = [[int(x / timestep) for x in l] for l in lengths_sec]

    return starts_seq, lengths_seq


def apply_exclusions(
    raw_pitch, starts_seq, lengths_seq, exclusion_functions, min_in_group
):
    for i in range(len(starts_seq)):
        these_seq = starts_seq[i]
        these_lens = lengths_seq[i]
        for j in range(len(these_seq))[::-1]:
            this_len = these_lens[j]
            this_start = these_seq[j]
            n_fails = 0
            for func in exclusion_functions:
                if func(raw_pitch[this_start : this_start + this_len]):
                    n_fails += 1
            if n_fails > 0:
                these_seq.pop(j)
                these_lens.pop(j)

    # minimum number in group to be pattern group
    starts_seq_exc = [seqs for seqs in starts_seq if len(seqs) >= min_in_group]
    lengths_seq_exc = [seqs for seqs in lengths_seq if len(seqs) >= min_in_group]

    return starts_seq_exc, lengths_seq_exc


def remove_below_length(starts_seq, lengths_seq, timestep, min_length):
    starts_seq_long = []
    lengths_seq_long = []
    for i, group in enumerate(lengths_seq):
        this_group_l = []
        this_group_s = []
        for j, l in enumerate(group):
            if l >= min_length / timestep:
                this_group_l.append(l)
                this_group_s.append(starts_seq[i][j])
        if this_group_s:
            starts_seq_long.append(this_group_s)
            lengths_seq_long.append(this_group_l)

    return starts_seq_long, lengths_seq_long


def extend_to_mask(starts_seq_exc, lengths_seq_exc, mask, toler=0.25):
    mask_i = list(range(len(mask)))
    starts_seq_ext = []
    lengths_seq_ext = []
    for i in range(len(starts_seq_exc)):
        s_group = starts_seq_exc[i]
        l_group = lengths_seq_exc[i]
        this_group_s = []
        this_group_l = []
        for j in range(len(s_group)):
            l = l_group[j]
            s1 = s_group[j]
            s2 = s1 + l

            s1_ = s1 - round(l * toler)
            s2_ = s2 + round(l * toler)

            midpoint = s1 + round(l / 2)

            s1_mask = list(mask[s1_:s1])
            s2_mask = list(mask[s2:s2_])
            s1_mask_i = list(mask_i[s1_:s1])
            s2_mask_i = list(mask_i[s2:s2_])

            if 1 in s1_mask:
                ix = len(s1_mask) - s1_mask[::-1].index(1) - 1
                s1 = s1_mask_i[ix]

            if 1 in s2_mask:
                ix = s2_mask.index(1)
                s2 = s2_mask_i[ix]

            l = s2 - s1

            this_group_s.append(s1)
            this_group_l.append(l)
        starts_seq_ext.append(this_group_s)
        lengths_seq_ext.append(this_group_l)

    return starts_seq_ext, lengths_seq_ext
