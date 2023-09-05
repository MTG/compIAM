import os

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.image
import matplotlib

# matplotlib.use('TkAgg')
import numpy as np
import numpy.ma as ma
from numpy.linalg import lstsq
import scipy.signal

import shutil
import seaborn as sns

from compiam.utils.pitch import pitch_seq_to_cents, pitch_to_cents
from compiam.utils import get_timestamp, myround, create_if_not_exists
from compiam.io import write_json


def plot_pitch(
    pitch,
    times,
    s_len=None,
    mask=None,
    yticks_dict=None,
    cents=False,
    tonic=None,
    emphasize=[],
    figsize=None,
    title=None,
    xlabel=None,
    ylabel=None,
    grid=True,
    ylim=None,
    xlim=None,
):
    """
    Plot graph of pitch over time

    :param pitch: Array of pitch values in Hz
    :type pitch: np.array
    :param times: Array of time values in seconds
    :type times: np.array
    :param s_len: If not None, take first <s_len> elements of <pitch> and <time> to plot
    :type s_len:  int
    :param mask: Array of bools indicating elements in <pitch> and <time> NOT to be plotted
    :type mask: np.array
    :param yticks_dict: Dict of {frequency name: frequency value (Hz)}
        ...if not None, yticks will be replaced in the relevant places with these names
    :type yticks_dict: dict(str, float)
    :param cents: Whether or not to convert frequency to cents above <tonic>
    :type cents: bool
    :param tonic: Tonic to make cent conversion in reference to - only relevant if <cents> is True.
    :type tonic: float
    :param emphasize: list of keys in yticks_dict to emphasize on plot (horizontal red line)
    :type emphasize: list(str)
    :param figsize: Tuple of figure size values
    :type figsize: tuple
    :param title: Title of figure, default None
    :type title: str
    :param xlabel: x axis label, default Time (s)
    :type xlabel: str
    :param ylabel: y axis label
        defaults to 'Cents Above Tonic of <tonic>Hz' if <cents>==True else 'Pitch (Hz)'
    :type ylabel: str
    :param grid: Whether to plot grid
    :type grid: bool
    :param ylim: Tuple of y limits, defaults to max and min in <pitch>
    :type ylim: bool
    :param xlim: Tuple of x limits, defaults to max and min in <time>
    :type xlim: bool

    :returns: Matplotlib objects for desired plot
    :rtype: (matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot)
    """
    if cents:
        assert tonic, "Cannot convert pitch to cents without reference <tonic>"
        p1 = pitch_seq_to_cents(pitch, tonic)
    else:
        p1 = pitch

    if mask is None:
        # If no masking required, create clear mask
        mask = np.full((len(pitch),), False)

    if s_len:
        assert s_len <= len(pitch), "Sample length is longer than length of pitch input"
    else:
        s_len = len(pitch)

    if figsize:
        assert isinstance(
            figsize, (tuple, list)
        ), "<figsize> should be a tuple of (width, height)"
        assert len(figsize) == 2, "<figsize> should be a tuple of (width, height)"
    else:
        figsize = (15, 5)

    if not xlabel:
        xlabel = "Time (s)"
    if not ylabel:
        ylabel = f"Cents Above Tonic of {round(tonic)}Hz" if cents else "Pitch (Hz)"

    pitch_masked = np.ma.masked_where(mask, p1)

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if grid:
        plt.grid()

    if xlim:
        xmin, xmax = xlim
    else:
        xmin = myround(min(times[:s_len]), 1)
        xmax = max(times[:s_len])

    if ylim:
        ymin, ymax = ylim
    else:
        sample = pitch_masked.data[:s_len]
        if not set(sample) == {None}:
            ymin_ = min([x for x in sample if x is not None])
            ymin = myround(ymin_, 50)
            ymax = max([x for x in sample if x is not None])
        else:
            ymin = 0
            ymax = 1000

    for s in emphasize:
        assert (
            yticks_dict
        ), "Empasize is for highlighting certain ticks in <yticks_dict>"
        if s in yticks_dict.values():
            pvals = [k for k, v in yticks_dict.items() if v == s]
            for p in pvals:
                p_ = pitch_to_cents(p, tonic) if cents else p
                ax.axhline(p_, color="#db1f1f", linestyle="--", linewidth=1)

    times_samp = times[:s_len]
    pitch_masked_samp = pitch_masked[:s_len]

    times_samp = times_samp[: min(len(times_samp), len(pitch_masked_samp))]
    pitch_masked_samp = pitch_masked_samp[
        : min(len(times_samp), len(pitch_masked_samp))
    ]
    plt.plot(times_samp, pitch_masked_samp, linewidth=0.7)

    if yticks_dict:
        tick_names = list(yticks_dict.values())
        tick_loc = [
            pitch_to_cents(p, tonic) if cents else p for p in yticks_dict.keys()
        ]
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(tick_names)

    ax.set_xticks(np.arange(xmin, xmax + 1, 1))

    plt.xticks(fontsize=8.5)
    ax.set_facecolor("#f2f2f2")

    ax.set_ylim((ymin, ymax))
    ax.set_xlim((xmin, xmax))

    if title:
        plt.title(title)

    return fig, ax


def plot_subsequence(sp, l, pitch, times, timestep, path=None, plot_kwargs={}):
    this_pitch = pitch[int(max(sp - l, 0)) : int(sp + 2 * l)]
    this_times = times[int(max(sp - l, 0)) : int(sp + 2 * l)]
    this_mask = this_pitch == 0

    fig, ax = plot_pitch(
        this_pitch,
        this_times,
        mask=this_mask,
        xlim=(min(this_times), max(this_times)),
        **plot_kwargs,
    )

    x_d = ax.lines[-1].get_xdata()
    y_d = ax.lines[-1].get_ydata()

    x = x_d[int(min(l, sp)) : int(l + min(l, sp))]
    y = y_d[int(min(l, sp)) : int(l + min(l, sp))]

    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    rect = Rectangle(
        (x_d[int(min(l, sp))], min_y),
        l * timestep,
        max_y - min_y,
        facecolor="lightgrey",
    )
    ax.add_patch(rect)

    ax.plot(x, y, linewidth=0.7, color="darkorange")
    ax.axvline(x=x_d[int(min(l, sp))], linestyle="dashed", color="black", linewidth=0.8)

    if path:
        plt.savefig(path, dpi=90)
        plt.close("all")
    else:
        return plt


def plot_subsequence_w_stability(
    sp, l, pitch, time, stable_mask, timestep, path=None, plot_kwargs={}
):
    this_pitch = pitch[int(max(sp - l, 0)) : int(sp + 2 * l)]
    this_times = time[int(max(sp - l, 0)) : int(sp + 2 * l)]
    this_stab = stable_mask[int(max(sp - l, 0)) : int(sp + 2 * l)]
    this_mask = this_pitch == 0

    fig, ax = plot_pitch(
        this_pitch,
        this_times,
        mask=this_mask,
        xlim=(min(this_times), max(this_times)),
        **plot_kwargs,
    )

    ax2 = ax.twinx()
    ax2.plot(this_times, this_stab, "g", linewidth=0.7, alpha=0.5, linestyle="--")

    x_d = ax.lines[-1].get_xdata()
    y_d = ax.lines[-1].get_ydata()

    x = x_d[int(min(l, sp)) : int(l + min(l, sp))]
    y = y_d[int(min(l, sp)) : int(l + min(l, sp))]

    max_y = ax.get_ylim()[1]
    min_y = ax.get_ylim()[0]
    rect = Rectangle(
        (x_d[int(min(l, sp))], min_y),
        l * timestep,
        max_y - min_y,
        facecolor="lightgrey",
    )
    ax.add_patch(rect)

    ax.plot(x, y, linewidth=0.7, color="darkorange")
    ax.axvline(x=x_d[int(min(l, sp))], linestyle="dashed", color="black", linewidth=0.8)

    if path:
        plt.savefig(path, dpi=90)
        plt.close("all")
    else:
        return plt


def plot_all_sequences(
    pitch, times, lengths, seq_list, direc, clear_dir=False, plot_kwargs={}
):
    timestep = times[1] - times[0]
    if clear_dir:
        try:
            shutil.rmtree(direc)
        except:
            pass
    for i, seqs in enumerate(seq_list):
        for si, s in enumerate(seqs):
            l = lengths[i][si]
            l_sec = round(lengths[i][0] * timestep, 1)
            t_sec = s * timestep
            str_pos = get_timestamp(t_sec)
            sp = int(s)
            plot_path = os.path.join(
                direc, f"motif_{i}_len={l_sec}/{si}_time={str_pos}.png"
            )
            create_if_not_exists(plot_path)
            plot_subsequence(
                sp, l, pitch, times, timestep, path=plot_path, plot_kwargs=plot_kwargs
            )


def add_line_to_plot(arr, x0, x1, y0, y1):
    length = int(np.hypot(x1 - x0, y1 - y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
    x = x.astype(int)
    y = y.astype(int)

    arr[x, y] = 100

    return arr


def get_lines(s1, s2):
    n = len(s1)
    all_lines = []
    for i in range(n):
        x0 = s1[i]
        x1 = s2[i]
        for j in range(n):
            if j == i:
                continue
            y0 = s1[j]
            y1 = s2[j]
            all_lines.append((x0, x1, y0, y1))
    return all_lines


def add_annotations_to_plot(arr, annotations, sr, cqt_window):
    arr_ = arr.copy()
    annotations_grouped = annotations.groupby("text")["s1"].apply(list).reset_index()

    annotations_grouped["s2"] = (
        annotations.groupby("text")["s2"].apply(list).reset_index()["s2"]
    )

    for i, row in annotations_grouped.iterrows():
        s1 = row["s1"]
        s2 = row["s2"]
        these_lines = get_lines(s1, s2)
        for x0, x1, y0, y1 in these_lines:
            arr_ = add_line_to_plot(
                arr_,
                int(x0 * sr / cqt_window),
                int(x1 * sr / cqt_window),
                int(y0 * sr / cqt_window),
                int(y1 * sr / cqt_window),
            )

    return arr_


def add_patterns_to_plot(arr, patterns, lengths, sr, cqt_window):
    arr_ = arr.copy()

    for i, group in enumerate(patterns):
        lens = lengths[i]

        s1 = group
        s2 = [g + lens[j] for j, g in enumerate(s1)]

        these_lines = get_lines(s1, s2)
        for x0, x1, y0, y1 in these_lines:
            arr_ = add_line_to_plot(
                arr_,
                int(x0 * sr / cqt_window),
                int(x1 * sr / cqt_window),
                int(y0 * sr / cqt_window),
                int(y1 * sr / cqt_window),
            )

    return arr_


def add_segments_to_plot(arr, segments):
    arr_ = arr.copy()
    for i, ((x0, y0), (x1, y1)) in enumerate(segments):
        arr_ = add_line_to_plot(arr_, int(x0), int(x1), int(y0), int(y1))
    return arr_


def join_plots(A, B, both_binary=True):
    h, w = A.shape

    rgb = np.zeros((h, w, 3))

    if both_binary:
        rgb[A > 0] = np.array([255, 255, 255])  # WHITE
        rgb[B > 0] = np.array([255, 0, 0])  # RED

        x, y = np.where(np.logical_and(A > 1, B > 1))
        rgb[x, y] = np.array([207, 126, 190])  # PINK
    else:
        scaled = ((A - A.min()) / (A.max() - A.min())) * 255
        rgb[:, :, 0] = scaled
        rgb[:, :, 1] = scaled
        rgb[:, :, 2] = scaled
        rgb[B > 0] = np.array([255, 0, 0])  # RED

    return rgb


def flush_matplotlib():
    plt.figure().clear()
    plt.close("all")
    plt.cla()
    plt.clf()


def sum_dist(d1, d2):
    """
    Sum two distributions, <d1> and <d2> along y axis

    :param d1,d2: numpy array of form [(x, y),...
    :type d1,d2: np.array

    :returns: one array with y values summed
    :rtype: np.array
    """
    if not d1.any():
        return d2
    elif not d2.any():
        return d1
    elif (not d1.shape[1] == 2) or (not d2.shape[1] == 2):
        raise ValueError("Arrays must have two columns")

    x1 = d1[:, 0]
    y1 = d1[:, 1]

    x2 = d2[:, 0]
    y2 = d2[:, 1]

    # get a sorted list of all x values
    x = np.unique(np.concatenate((x1, x2)))

    # interpolate y1 and y2 on the combined x values
    yi1 = np.interp(x, x1, y1, left=0, right=0)
    yi2 = np.interp(x, x2, y2, left=0, right=0)

    y = yi1 + yi2

    return np.array([x, y]).T


def get_kde_no_plot(p, bw):
    """
    Get kde of <p> for bandwidth <bw> using
    seaborn without plotting (just returning values)
    """
    # remove any plots from matplotlib backend
    flush_matplotlib()

    # Is matplotlib in interactive mode?
    interactive = matplotlib.is_interactive()

    # turn interactive mode off
    plt.ioff()

    # get data
    ph = sns.kdeplot(p, bw_method=bw)

    data = ph.lines[0].get_xydata()

    # remove plots from matplotlib backend
    flush_matplotlib()

    # if interactive mode was on, turn it back on
    if interactive:
        plt.ion()

    return data


def line_through_points(x0, y0, x1, y1):
    """
    Return two functions corresponding to straight line learnt
    via least squares regression between two points (x0,y0)
    and (x1, y1). These functions retrieve x and y values
    from that line.

    :param x0,y0,x1,y1: coordinates values corresponding to two points
    :type x0,y0,x1,y1: float

    :returns: Two lambda functions, get_x and get_y that takes as a value, y or x
        respectively and returns x or y values along learnt line
    :rtype: two python functions
    """
    centroids = [(x0, y0), (x1, y1)]
    x_coords, y_coords = zip(*centroids)

    # gradient and intercecpt of line passing through centroids
    A = np.vstack([x_coords, np.ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]

    # functions for converting between
    # x and y on that line
    get_y = lambda xin: m * xin + c
    get_x = lambda yin: (yin - c) / m

    return get_x, get_y


def pitch_density_single_octave(p, bw, n_octaves=2):
    """
    Get pitch density folded to a single octave for
    pitch time series, <p>

    :param p: array of time series pitch values, must be in cents
    :type p: np.array
    :param bw: Bandwidth of gaussian kernel
    :type bw: float
    :param n_octaves: Number of octaves below and above tonic (0 cents) to search over, default 3
    :type n_octaves: int
    """
    flush_matplotlib()
    data = get_kde_no_plot(p, bw)

    # Split data into octaves
    # and sum
    summed_d = np.empty((0, 2))
    for i in np.arange(-n_octaves * 1200, n_octaves * 1200, 1200):
        mask = (data[:, 0] >= i) & (data[:, 0] < i + 1200)
        d = data[mask]

        # If there is activity in this octave
        # Ensure all lines begin and end at 0 and 1200
        if d.any():
            # Force to same octave
            d[:, 0] = d[:, 0] % 1200

            # only include if there are sufficient data points
            if len(d) > 8:
                # line through first 3 points
                # (least squares)
                if d[:, 0].min() > 0:
                    first_point = d[0]
                    second_point = d[3]
                    getx_min, gety_min = line_through_points(
                        first_point[0], first_point[1], second_point[0], second_point[1]
                    )

                    # new point to append at x=0
                    new_point1 = np.array([0, max([0, gety_min(0)])])

                    # append at beginning
                    d = np.concatenate([[new_point1], d])

                if d[:, 0].max() < 1200:
                    # line through final 3 points
                    # (least squares)
                    final_point = d[-1]
                    penultimate_point = d[-3]
                    gety_max, gety_max = line_through_points(
                        final_point[0],
                        final_point[1],
                        penultimate_point[0],
                        penultimate_point[1],
                    )

                    # new point to append after all others
                    new_point2 = np.array([1200, max([0, gety_max(1200)])])

                    # append at end
                    d = np.concatenate([d, [new_point2]])

                # add to sum
                summed_d = sum_dist(summed_d, d)

    # normalise area under curve to equal 1
    current_area = np.trapz(summed_d[:, 1])
    summed_d[:, 1] /= current_area

    return summed_d


def get_pitch_density(
    pitch,
    bw=0.05,
    annotations={},
    title="",
    title_fontsize=12,
    figsize=(16, 6),
    ytick_size=None,
    xlim=None,
    cents=False,
    octave=False,
    imode=True,
):
    flush_matplotlib()
    interactive = matplotlib.is_interactive()
    if not imode:
        plt.ioff()

    # get relevant pitch values
    p = np.copy(pitch)
    if cents:
        # if cents remove None values (corresponding to silence)
        p = p[p != np.array(None)]
        units = "cents"
    else:
        if None in pitch:
            raise ValueError(
                "Nones in <pitch>, if <pitch> is in cents, please specify cents=True"
            )
        # If fequency remove 0s (corresponding to silence)
        p = p[p != 0]
        units = "Hz"

    # get data either folded/unfolded
    if octave:
        if not units == "cents":
            raise ValueError("Octave functionality only available with cents input")
        xlim = (0, 1200)
        if max(p) <= 1200 and min(p) >= 0:
            # No need to fold, all data in one octave
            data = get_kde_no_plot(p, bw)
        else:
            data = pitch_density_single_octave(p, bw)
    else:
        data = get_kde_no_plot(p, bw)

    # plot setup
    plt.figure(figsize=figsize)
    plt.title(title, fontsize=title_fontsize)
    plt.grid()

    ax = plt.gca()
    ax.set_facecolor("#f2f2f2")

    if ytick_size:
        plt.yticks(fontsize=xtick_size)
    else:
        # turn off y ticks
        plt.yticks([], [])

    plt.ylabel("Density")
    plt.xlabel(f"Pitch ({units})")

    if xlim:
        plt.xlim(xlim)

    plt.plot(data[:, 0], data[:, 1])
    splt = plt.gca()

    y_max = splt.get_ylim()[1]
    x_min, x_max = splt.get_xlim()

    for k, v in annotations.items():
        if x_min <= k <= x_max:
            plt.axvline(k, color="firebrick", linestyle="--", linewidth=1)
            plt.text(
                k, y_max - y_max * 0.1, v, color="firebrick", fontsize=12, rotation=270
            )

    # remove all plots on backend
    flush_matplotlib()

    # turn interactive mode back on, if it was enabled
    if interactive:
        plt.ion()

    return splt
