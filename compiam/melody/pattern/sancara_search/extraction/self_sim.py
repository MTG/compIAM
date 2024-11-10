import os
import shutil
import skimage
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.signal import convolve2d

from compiam.melody.pattern.sancara_search.extraction.img import (
    remove_diagonal,
    convolve_array,
    binarize,
    diagonal_gaussian,
    apply_bin_op,
    make_symmetric,
    edges_to_contours,
)
from compiam.utils import create_if_not_exists, run_or_cache

from compiam.melody.pattern.sancara_search.extraction.sequence import (
    convert_seqs_to_timestep,
    remove_below_length,
    add_center_to_mask,
)
from compiam.melody.pattern.sancara_search.extraction.evaluation import (
    evaluate,
    get_coverage,
    get_grouping_accuracy,
)
from compiam.melody.pattern.sancara_search.extraction.visualisation import (
    plot_all_sequences,
    plot_pitch,
    flush_matplotlib,
)
from compiam.melody.pattern.sancara_search.extraction.io import (
    load_sim_matrix,
    write_all_sequence_audio,
    load_pkl,
    write_pkl,
)
from compiam.melody.pattern.sancara_search.extraction.pitch import (
    cents_to_pitch,
    pitch_seq_to_cents,
    pitch_to_cents,
    get_timeseries,
    interpolate_below_length,
)
from compiam.melody.pattern.sancara_search.extraction.segments import (
    line_through_points,
    trim_silence,
    break_all_segments,
    remove_short,
    extend_segments,
    join_all_segments,
    extend_groups_to_mask,
    group_segments,
    group_overlapping,
    group_by_distance,
    trim_silence,
    segments_from_matrix,
    remove_group_duplicates,
)
from compiam.utils import get_logger

logger = get_logger(__name__)


def self_similarity(
    features, exclusion_mask=None, timestep=None, hop_length=None, sr=44100
):
    """
    Compute self similarity matrix between features in <features>. If an <exclusion_mask>
    is passed. Regions corresponding to that mask will be excluded from the computation and
    the returned matrix will correspond only to those regions marked as 0 in the mask.

    :param features: array of features extracted from audio
    :type features: np.ndarray
    :param exclusion_mask: array of 0 and 1, should be masked or not? [Optional]
    :type exclusion_mask: np.ndarray or None
    :param timestep: time in seconds between elements of <exclusion_mask>
        Only required if <exclusion_mask> is passed
    :type timestep: float or None
    :param hop_length: number of audio frames corresponding to one element in <features>
        Only required if <exclusion_mask> is passed
    :type hop_length: int or None
    :param sr: sampling rate of audio corresponding to <features>
        Only required if <exclusion_mask> is passed
    :type sr: int or None

    :returns:
        if exclusion mask is passed return...
            matrix - self similarity matrix
            orig_sparse_lookup - dict of {index in orig array: index of same element in sparse array}
            sparse_orig_lookup - dict of {index in sparse array: index of same element in orig array}
            boundaries_orig - list of boundaries between wanted and unwanted regions in orig array
            boundaries_sparse -  list of boundaries between formally separated wanted regions in sparse array
        else return
            matrix - self similarity matrix
    :rtype: (np.ndarray, dict, dict, list, list) or np.ndarray
    """
    em = not (exclusion_mask is None)
    if em:
        assert all(
            [not timestep is None, not hop_length is None, not sr is None]
        ), "To use exclusion mask, <timestep>, <hop_length> and <sr> must also be passed"

    # Deal with masking if any
    if em:
        features_mask = convert_mask(features, exclusion_mask, timestep, hop_length, sr)
        (
            orig_sparse_lookup,
            sparse_orig_lookup,
            boundaries_orig,
            boundaries_sparse,
        ) = get_conversion_mappings(features_mask)
    else:
        orig_sparse_lookup = None
        sparse_orig_lookup = None
        boundaries_orig = None
        boundaries_sparse = None

    # Indices we want to keep
    good_ix = np.where(features_mask == 0)[0]

    # Compute self similarity
    sparse_features = features[good_ix]
    matrix = create_ss_matrix(sparse_features)

    # Normalise self similarity matrix
    matrix_norm = normalise_self_sim(matrix)

    if em:
        return (
            matrix_norm,
            orig_sparse_lookup,
            sparse_orig_lookup,
            boundaries_orig,
            boundaries_sparse,
        )
    else:
        return matrix_norm


def convert_mask(arr, mask, timestep, hop_length, sr):
    """
    Get mask of excluded regions in the same dimension as array, <arr>

    :param arr: array corresponding to features extracted from audio
    :type arr: np.ndarray
    :param mask: Mask indicating whether element should be excluded (different dimensions to <arr>)
    :type mask: np.ndarray
    :param timestep: time in seconds between each element in <mask>
    :type timestep: float
    :param hop_length: how many frames of audio correspond to each element in <arr>
    :type hop_length: int
    :param sr: sampling rate of audio from which <arr> was computed
    :type sr: int

    :returns: array of mask values equal in length to one dimension of <arr> - 0/1 is masked?
    :rtype: np.ndarray
    """
    # get mask of silent and stable regions
    new_mask = []
    for i in range(arr.shape[0]):
        # what is the time at this element of arr?
        t = (i + 1) * hop_length / sr
        # find index in mask
        ix = round(t / timestep)
        # append mask value for this point
        new_mask.append(mask[ix])
    return np.array(new_mask)


def get_conversion_mappings(mask):
    """
    Before reducing an array to only include elements that do not correspond
    to <mask>. We want to record the relationship between the new (sparse) array
    index and the old (orig) array.

    :param mask: mask of 0/1 - is element to be excluded
    :param type: np.ndarray

    :returns:
        orig_sparse_lookup - dict of {index in orig array: index of same element in sparse array}
        sparse_orig_lookup - dict of {index in sparse array: index of same element in orig array}
        boundaries_orig - list of boundaries between wanted and unwanted regions in orig array
        boundaries_sparse -  list of boundaries between formally separated wanted regions in sparse array
    :rtype: (dict, dict, list, list)
    """
    # Indices we want to keep
    good_ix = np.where(mask == 0)[0]

    # Mapping between old and new indices
    orig_sparse_lookup = {g: s for s, g in enumerate(good_ix)}
    sparse_orig_lookup = {s: g for g, s in orig_sparse_lookup.items()}

    # Indices corresponding to boundaries
    # between wanted and unwanted regions
    # in original array
    boundaries_orig = []
    for i in range(1, len(mask)):
        curr = mask[i]
        prev = mask[i - 1]
        if curr == 0 and prev == 1:
            boundaries_orig.append(i)
        elif curr == 1 and prev == 0:
            boundaries_orig.append(i - 1)

    # Boundaries corresponding to newly joined
    # regions in sparse array
    boundaries_sparse = np.array([orig_sparse_lookup[i] for i in boundaries_orig])

    # Boundaries contain two consecutive boundaries for each gap
    # but not if the excluded region leads to the end of the track
    red_boundaries_sparse = []
    boundaries_mask = [0] * len(boundaries_sparse)
    for i in range(len(boundaries_sparse)):
        if i == 0:
            red_boundaries_sparse.append(boundaries_sparse[i])
            boundaries_mask[i] = 1
        if boundaries_mask[i] == 1:
            continue
        curr = boundaries_sparse[i]
        prev = boundaries_sparse[i - 1]
        if curr - prev == 1:
            red_boundaries_sparse.append(prev)
            boundaries_mask[i] = 1
            boundaries_mask[i - 1] = 1
        else:
            red_boundaries_sparse.append(curr)
            boundaries_mask[i] = 1
    boundaries_sparse = np.array(sorted(list(set(red_boundaries_sparse))))

    return orig_sparse_lookup, sparse_orig_lookup, boundaries_orig, boundaries_sparse


def create_ss_matrix(feats, mode="cosine"):
    """
    Compute self similarity matrix between features in <feats>
    using distance measure, <mode>

    :param feats: array of features
    :type feats: np.ndarray
    :param mode: name of distance measure (recognised by scipy.spatial.distance)
    :type mode: str

    :returns: self similarity matrix
    :rtype: np.ndarray
    """
    matrix = squareform(pdist(np.vstack(feats.detach().numpy()), metric=mode))
    return matrix


def normalise_self_sim(matrix):
    """
    Normalise self similarity matrix:
        invert and convolve

    :param matrix: self similarity matrix
    :type matrix: np.ndarray

    :returns: matrix normalized, same dimensions
    :rtype: np.ndarray
    """
    matrix = 1 / (matrix + 1e-6)

    for k in range(-8, 9):
        eye = 1 - np.eye(*matrix.shape, k=k)
        matrix = matrix * eye

    flength = 10
    ey = np.eye(flength) + np.eye(flength, k=1) + np.eye(flength, k=-1)
    matrix = convolve2d(matrix, ey, mode="same")

    diag_mask = np.ones(matrix.shape)
    diag_mask = (diag_mask - np.diag(np.ones(matrix.shape[0]))).astype(bool)

    mat_min = np.min(matrix[diag_mask])
    mat_max = np.max(matrix[diag_mask])

    matrix[~diag_mask] = 0

    matrix = zero_normalise(matrix)

    return matrix


def zero_normalise(matrix):
    matrix = matrix - matrix.min()
    matrix = matrix / (matrix.max() + 1e-8)
    return matrix


def get_report_paths(out_dir):
    """
    Get dictionary of fielpaths relevant to progress plots
    in extract_segments()

    :params out_dir: directory path to save plots in
    :type out_dir: str

    :returns: dict of filepaths
    :rtype: dict
    """
    sim = os.path.join(out_dir, "1_simsave.png") if out_dir else None
    conv = os.path.join(out_dir, "2_conv.png") if out_dir else None
    binar = os.path.join(out_dir, "3_binary.png") if out_dir else None
    diag = os.path.join(out_dir, "4_diag.png") if out_dir else None
    gauss = os.path.join(out_dir, "5_gauss.png") if out_dir else None
    cont = os.path.join(out_dir, "6_cont.png") if out_dir else None
    close = os.path.join(out_dir, "6_close.png") if out_dir else None
    binop = os.path.join(out_dir, "7_binop.png") if out_dir else None

    return {
        "sim": sim,
        "conv": conv,
        "binar": binar,
        "diag": diag,
        "gauss": gauss,
        "cont": cont,
        "close": close,
        "binop": binop,
    }


def save_matrix(X, filepath):
    """
    if <filepath>, save <X> at <filepath>

    :param X:  matrix to save
    :type X: np.ndarray
    :param filepath: filepath
    :type filepath: str or None
    """
    if filepath:
        create_if_not_exists(filepath)
        skimage.io.imsave(filepath, X)


def get_param_hash_filepath(out_dir, *params):
    """
    Build filepath by creating string of input <params>
    in <out_dir>

    :params out_dir: directory path
    :type out_dir: str
    :params params: arguments, any type
    :type params: arguments, any type

    :returns: filepath unique to input params in <out_dir>
    :rtype: str
    """
    if out_dir is None:
        return None
    param_hash = str(params)
    return os.path.join(out_dir, f"{param_hash}.pkl")


def sparse_to_original(all_segments, boundaries_sparse, lookup):
    """
    Convert indices corresponding to segments in <all_segments>
    to their non-sparse form using mapping in <lookup>

    :param all_segments:  list of segments, [(x0,y0),(x1,y1),...]
    :type all_segments: list
    :param boundaries_sparse: list indices in sparse array corresponding to splits in original array
    :type boundaries_sparse: list
    :param lookup: dict of sparse_index:non-sparse index
    :type lookup: dict

    :returns: <all_segments> with indices replaced according to lookup
    :rtype: list
    """
    boundaries_sparse = [x for x in boundaries_sparse if x != 0]
    all_segments_scaled_x = []
    for seg in all_segments:
        ((x0, y0), (x1, y1)) = seg
        get_x, get_y = line_through_points(x0, y0, x1, y1)

        boundaries_in_x = sorted([i for i in boundaries_sparse if i >= x0 and i <= x1])
        current_x0 = x0
        if boundaries_in_x:
            for b in boundaries_in_x:
                x0_ = current_x0
                x1_ = b - 1
                y0_ = int(get_y(x0_))
                y1_ = int(get_y(x1_))
                all_segments_scaled_x.append(((x0_, y0_), (x1_, y1_)))
                current_x0 = b + 1

            if current_x0 > x1:
                x0_ = current_x0
                x1_ = x1
                y0_ = int(get_y(x0_))
                y1_ = int(get_y(x1_))
                all_segments_scaled_x.append(((x0_, y0_), (x1_, y1_)))
        else:
            all_segments_scaled_x.append(((x0, y0), (x1, y1)))

    all_segments_scaled_x_reduced = remove_short(all_segments_scaled_x, 1)

    all_segments_scaled = []
    for seg in all_segments_scaled_x_reduced:
        ((x0, y0), (x1, y1)) = seg
        get_x, get_y = line_through_points(x0, y0, x1, y1)

        boundaries_in_y = sorted([i for i in boundaries_sparse if i >= y0 and i <= y1])
        current_y0 = y0
        if boundaries_in_y:
            for b in boundaries_in_y:
                y0_ = current_y0
                y1_ = b - 1
                x0_ = int(get_x(y0_))
                x1_ = int(get_x(y1_))
                all_segments_scaled.append(((x0_, y0_), (x1_, y1_)))
                current_y0 = b + 1

            if current_y0 < y1:
                y0_ = current_y0
                y1_ = y1
                x0_ = int(get_x(y0_))
                x1_ = int(get_x(y1_))
                all_segments_scaled.append(((x0_, y0_), (x1_, y1_)))
        else:
            all_segments_scaled.append(((x0, y0), (x1, y1)))

    all_segments_scaled_reduced = remove_short(all_segments_scaled, 1)

    all_segments_converted = []

    de = 0

    for i, seg in enumerate(all_segments_scaled_reduced):
        ((x0, y0), (x1, y1)) = seg
        while (
            (x0 in boundaries_sparse)
            or (x1 in boundaries_sparse)
            or (y0 in boundaries_sparse)
            or (y1 in boundaries_sparse)
        ):
            if x0 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                x0 = x0 + 1
                x1 = x1
                y0 = round(get_y(x0))
                y1 = round(get_y(x1))

            if x1 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                x0 = x0
                x1 = x1 - 1
                y0 = round(get_y(x0))
                y1 = round(get_y(x1))

            if y0 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                y0 = y0 + 1
                y1 = y1
                x0 = round(get_x(y0))
                x1 = round(get_x(y1))

            if y1 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                y0 = y0
                y1 = y1 - 1
                x0 = round(get_x(y0))
                x1 = round(get_x(y1))

        x0_ = lookup[x0 + de]
        y0_ = lookup[y0 + de]
        x1_ = lookup[x1 + de]
        y1_ = lookup[y1 + de]
        all_segments_converted.append(((x0_, y0_), (x1_, y1_)))

    return all_segments_converted


def zero_norm_matrix(X):
    X = X - X.min()
    X /= X.max() + 1e-8
    return X


class segmentExtractor:
    """
    Manipulate and extract segments from self similarity matrix
    """

    def __init__(self, X, window_size, sr=44100, cache_dir=None):
        self.X = X
        self.shape = X.shape
        self.window_size = window_size
        self.sr = sr
        self.cache_dir = cache_dir

        # initialise arrays
        self.X_conv = None
        self.X_bin = None
        self.X_diag = None
        self.X_gauss = None
        self.X_cont = None
        self.X_sym = None
        self.X_fill = None
        self.X_binop = None
        self.X_proc = None

        # Initialize status
        self.emphasized = False
        self.extracted = False

        # cache paths
        self._cache_base = os.path.join(cache_dir, "{0}", "") if cache_dir else None
        self._segment_convolve_cache = (
            self._cache_base.format("convolve") if cache_dir else None
        )
        self._segment_cache = self._cache_base.format("segments") if cache_dir else None
        self._segment_ext_cache = (
            self._cache_base.format("segments_extended") if cache_dir else None
        )
        self._segment_join_cache = (
            self._cache_base.format("segments_joined") if cache_dir else None
        )
        self._segment_group_cache = (
            self._cache_base.format("segments_groups") if cache_dir else None
        )
        self._segment_group_overlap_cache = (
            self._cache_base.format("segment_overlap") if cache_dir else None
        )

    def emphasize_diagonals(
        self,
        bin_thresh=0.025,
        gauss_sigma=None,
        cont_thresh=None,
        etc_kernel_size=10,
        binop_dim=3,
        image_report=False,
        verbose=False,
    ):
        """
        From self similarity matrix, self.X. Emphasize diagonals using a series
        of image processing steps.

        :param bin_thresh: Threshold for binarization of self similarity array.
            Values below this threshold are set to 0 (not significant), those
            above or equal too are set to 1. Very important parameter
        :type bin_thresh: float
        :param gauss_sigma: If not None, sigma for diagonal gaussian blur to
            apply to matrix
        :type gauss_sigma: float or None
        :param cont_thresh: Only applicable if <gauss_sigma>. This binary
            threshold isreapplied after gaussian blur to ensure matrix of
            0 and 1. if None, equal to <bin_thresh>
        :type cont_thresh: float or None
        :param etc_kernel_size: Kernel size for morphological closing
        :type etc_kernel_size: int
        :param binop_dim: square dimension of binary opening structure
            (square matrix of zeros with 1 across the diagonal)
        :type binop_dim: int
        :param image_report: str corresponding to folder to save progress images in.
        :type image_report: None
        :param verbose: Display progress
        :type verbose: bool

        :returns: list of segments in the form [((x0,y0),(x1,y1)),..]
        :rtype: list
        """
        self.bin_thresh = bin_thresh
        self.gauss_sigma = gauss_sigma
        self.etc_kernel_size = etc_kernel_size
        self.binop_dim = binop_dim
        self.image_report = image_report

        self.report_fns = get_report_paths(image_report)

        # Save original self similarity matrix
        save_matrix(self.X, self.report_fns["sim"])

        ####################
        ## Convert params ##
        ####################
        self.cont_thresh = self.bin_thresh if not cont_thresh else cont_thresh

        #########################
        ## Emphasize Diagonals ##
        #########################
        if verbose:
            logger.info("Convolving similarity matrix")
        self.conv_path = get_param_hash_filepath(
            self._segment_convolve_cache,
        )
        self.X_conv = run_or_cache(convolve_array, [self.X], self.conv_path)
        self.X_conv = zero_norm_matrix(self.X_conv)
        save_matrix(self.X_conv, self.report_fns["conv"])

        if verbose:
            logger.info("Binarizing convolved array")
        self.X_bin = binarize(self.X_conv, self.bin_thresh)
        save_matrix(self.X_bin, self.report_fns["binar"])

        if verbose:
            logger.info("Removing diagonal")
        self.X_diag = remove_diagonal(self.X_bin)
        save_matrix(self.X_diag, self.report_fns["diag"])

        if self.gauss_sigma:
            if verbose:
                logger.info("Applying diagonal gaussian filter")
            self.X_gauss = diagonal_gaussian(self.X_diag, self.gauss_sigma)
            save_matrix(self.X_gauss, self.report_fns["gauss"])

            if verbose:
                logger.info("Binarize gaussian blurred similarity matrix")
            self.X_cont = binarize(self.X_gauss, self.cont_thresh)
            save_matrix(self.X_cont, self.report_fns["cont"])
        else:
            self.X_gauss = self.X_diag
            self.X_cont = self.X_gauss

        if verbose:
            logger.info("Ensuring symmetry between upper and lower triangle in array")
        self.X_sym = make_symmetric(self.X_cont)

        if verbose:
            logger.info("Identifying and isolating regions between edges")
        self.X_fill = edges_to_contours(self.X_sym, self.etc_kernel_size)
        save_matrix(self.X_fill, self.report_fns["close"])

        if verbose:
            logger.info(
                "Cleaning isolated non-directional regions using morphological opening"
            )
        self.X_binop = apply_bin_op(self.X_fill, self.binop_dim)

        if verbose:
            logger.info("Ensuring symmetry between upper and lower triangle in array")
        self.X_proc = make_symmetric(self.X_binop)
        save_matrix(self.X_proc, self.report_fns["binop"])

        self.emphasized = True

        return self.X_proc

    def extract_segments(
        self,
        etc_kernel_size=10,
        binop_dim=3,
        perc_tail=0.5,
        bin_thresh_segment=None,
        min_diff_trav=0.5,
        min_pattern_length_seconds=2,
        boundaries=None,
        lookup=None,
        break_mask=None,
        timestep=None,
        verbose=False,
    ):
        """
        From self similarity matrix, <self.X_proc>. Return list of segments,
        each corresponding to two regions of the input axis.

        :param etc_kernel_size: Kernel size for morphological closing
        :type etc_kernel_size: int
        :param binop_dim: square dimension of binary opening structure
            (square matrix of zeros with 1 across the diagonal)
        :type binop_dim: int
        :param perc_tail: Percentage either size of a segment along its
            trajectory considered for lower threshold for significance
        :type perc_tail: int
        :param bin_thresh_segment: Reduced <bin_thresh> threshold for
            areas neighbouring identified segments. If None, use 0.5*<bin_thresh>
        :type bin_thresh_segment: float
        :param min_diff_trav: Min time difference in seconds between
            two segments for them to be joined to one.
        :type min_diff_trav: float
        :param min_pattern_length_seconds: Minimum length of any
            returned pattern in seconds
        :type min_pattern_length_seconds: float
        :param boundaries: list of boundaries in <X> corresponding
            to breaks due to sparsity
        :type boundaries: list or None
        :param lookup: Lookup of sparse index (in X): non-sparse index
        :type lookup: dict
        :param break_mask: any segment that traverses a non-zero element
            in <break_mask> is broken into two according to this non-zero value
        :type break_mask: array
        :param timestep: Time in seconds between each element in <break_mask>
        :type timestep: float or None
        :param verbose: Display progress
        :type verbose: bool

        :returns: list of segments in the form [((x0,y0),(x1,y1)),..]
        :rtype: list
        """
        ############
        ## Checks ##
        ############
        if not self.emphasized:
            raise Exception(
                "Please run self.emphasize_diagonals before attempting to extract segments."
            )

        if break_mask is not None:
            assert (
                timestep is not None
            ), "If <break_mask> is passed, timestep too must be specified"

        if boundaries is not None:
            assert (
                lookup is not None
            ), "If <boundaries> is passed, lookup too must be specified"

        ############
        ## Params ##
        ############
        self.min_diff_trav = min_diff_trav
        # in terms of elements matrix elements
        self.min_length_cqt = min_pattern_length_seconds * self.sr / self.window_size
        # translate min_diff_trav to corresponding diagonal distance
        self.min_diff_trav_hyp = (2 * min_diff_trav**2) ** 0.5
        self.min_diff_trav_seq = self.min_diff_trav_hyp * self.sr / self.window_size
        self.bin_thresh_segment = (
            self.bin_thresh * 0.5 if not bin_thresh_segment else bin_thresh_segment
        )
        self.perc_tail = perc_tail
        self.min_pattern_length_seconds = min_pattern_length_seconds
        self.boundaries = boundaries
        self.lookup = lookup
        self.break_mask = break_mask
        self.timestep = timestep

        ######################
        ## Extract segments ##
        ######################
        if verbose:
            logger.info("Extracting segments")
        self.seg_path = get_param_hash_filepath(
            self._segment_cache,
            self.bin_thresh,
            self.gauss_sigma,
            self.cont_thresh,
            self.etc_kernel_size,
            self.binop_dim,
        )
        self.all_segments = run_or_cache(
            segments_from_matrix, [self.X_bin], self.seg_path
        )

        if verbose:
            logger.info("Extending Segments")
        self.seg_ext_path = get_param_hash_filepath(
            self._segment_ext_cache,
            self.bin_thresh,
            self.gauss_sigma,
            self.cont_thresh,
            self.etc_kernel_size,
            self.binop_dim,
            self.perc_tail,
            self.bin_thresh_segment,
        )
        args = [
            self.all_segments,
            self.X_sym,
            self.X_conv,
            self.perc_tail,
            self.bin_thresh_segment,
        ]
        self.all_segments_extended = run_or_cache(
            extend_segments, args, self.seg_ext_path
        )

        if verbose:
            logger.info(f"    {len(self.all_segments_extended)} extended segments...")

        self.all_segments_extended_reduced = remove_short(self.all_segments_extended, 1)

        if verbose:
            logger.info("Converting sparse segment indices to original")
        if not self.boundaries is None:
            self.all_segments_converted = sparse_to_original(
                self.all_segments_extended_reduced, self.boundaries, self.lookup
            )
        else:
            self.all_segments_converted = self.all_segments_extended_reduced

        if verbose:
            logger.info("Joining segments that are sufficiently close")
        self.seg_join_path = get_param_hash_filepath(
            self._segment_join_cache,
            self.bin_thresh,
            self.gauss_sigma,
            self.cont_thresh,
            self.etc_kernel_size,
            self.binop_dim,
            self.perc_tail,
            self.bin_thresh_segment,
            self.min_diff_trav_seq,
        )
        args = [self.all_segments_converted, self.min_diff_trav_seq]
        self.all_segments_joined = run_or_cache(
            join_all_segments,
            [self.all_segments_converted, self.min_diff_trav_seq],
            self.seg_join_path,
        )
        if verbose:
            logger.info(f"    {len(self.all_segments_joined)} joined segments...")

        if verbose:
            logger.info("Breaking segments with silent/stable regions")
        if not self.break_mask is None:
            self.all_broken_segments = break_all_segments(
                self.all_segments_joined,
                self.break_mask,
                self.window_size,
                self.sr,
                self.timestep,
            )
        else:
            self.all_broken_segments = self.all_segments_joined
        if verbose:
            logger.info(f"    {len(self.all_broken_segments)} broken segments...")

        if verbose:
            logger.info("Reducing Segments")
        self.all_segments_reduced = remove_short(
            self.all_broken_segments, self.min_length_cqt
        )
        if verbose:
            logger.info(
                f"    {len(self.all_segments_reduced)} segments above minimum length of {self.min_pattern_length_seconds}s..."
            )

        self.extracted = True

        return self.all_segments_reduced

    def group_segments(
        self,
        all_segments,
        break_mask,
        pitch,
        ext_mask_tol=0.5,
        match_tol=1,
        dupl_perc_overlap_inter=0.9,
        dupl_perc_overlap_intra=0.55,
        group_len_var=1.0,
        n_dtw=10,
        thresh_dtw=10,
        thresh_cos=None,
        min_pattern_length_seconds=2,
        min_in_group=2,
        verbose=False,
    ):
        ############
        ## Params ##
        ############
        self.pitch = pitch
        break_mask = break_mask
        self.ext_mask_tol = ext_mask_tol
        self.match_tol = match_tol
        self.dupl_perc_overlap_inter = dupl_perc_overlap_inter
        self.dupl_perc_overlap_intra = dupl_perc_overlap_intra
        self.group_len_var = group_len_var
        self.n_dtw = n_dtw
        self.thresh_dtw = thresh_dtw
        self.thresh_cos = thresh_cos
        self.min_pattern_length_seconds = min_pattern_length_seconds
        self.min_in_group = min_in_group

        if verbose:
            logger.info("Identifying Segment Groups")
        self.group_path = get_param_hash_filepath(
            self._segment_group_cache,
            self.bin_thresh,
            self.gauss_sigma,
            self.cont_thresh,
            self.etc_kernel_size,
            self.binop_dim,
            self.perc_tail,
            self.bin_thresh_segment,
            self.min_diff_trav_seq,
            self.min_length_cqt,
            self.match_tol,
        )
        args = [
            all_segments,
            self.min_length_cqt,
            self.match_tol,
            break_mask,
            self.window_size,
            self.timestep,
            self.sr,
            self.pitch,
        ]
        all_groups = run_or_cache(group_segments, args, self.group_path)

        if verbose:
            logger.info("Extending segments to silence/stability")
        all_groups_ext = extend_groups_to_mask(
            all_groups,
            break_mask,
            self.window_size,
            self.sr,
            self.timestep,
            toler=self.ext_mask_tol,
        )

        if verbose:
            logger.info("Trimming Silence")

        all_groups_sil = trim_silence(
            all_groups_ext, self.pitch, self.window_size, self.sr, self.timestep
        )

        all_groups_sil = [[(i, j) for i, j in x if j > i] for x in all_groups_sil]

        all_groups_sil = [
            remove_group_duplicates(g, self.dupl_perc_overlap_intra)
            for g in all_groups_sil
        ]

        if verbose:
            logger.info("Identifying Segment Groups")
        self.segment_overlap_path = get_param_hash_filepath(
            self._segment_group_overlap_cache,
            self.bin_thresh,
            self.gauss_sigma,
            self.cont_thresh,
            self.etc_kernel_size,
            self.binop_dim,
            self.perc_tail,
            self.bin_thresh_segment,
            self.min_diff_trav_seq,
            self.min_length_cqt,
            self.match_tol,
            self.dupl_perc_overlap_inter,
            self.group_len_var,
        )
        all_groups = run_or_cache(
            group_overlapping,
            [all_groups_sil, self.dupl_perc_overlap_inter, self.group_len_var],
            self.segment_overlap_path,
        )

        if self.thresh_dtw:
            if verbose:
                logger.info("Joining geometrically close groups using pitch tracks")
            all_groups_dtw = group_by_distance(
                all_groups,
                self.pitch,
                self.n_dtw,
                self.thresh_dtw,
                self.thresh_cos,
                self.group_len_var,
                self.window_size,
                self.sr,
                self.timestep,
            )
            if verbose:
                logger.info(f"    {len(all_groups_dtw)} groups after join...")
        else:
            all_groups_dtw = all_groups

        # all_groups_over = group_overlapping(all_groups_dtw, 0.1, group_len_var)
        all_groups_rgd = [
            remove_group_duplicates(g, self.dupl_perc_overlap_intra)
            for g in all_groups_dtw
        ]

        if verbose:
            logger.info("Grouping overlapping")
        all_groups_dov = group_overlapping(
            all_groups_rgd, self.dupl_perc_overlap_inter, self.group_len_var
        )
        if verbose:
            logger.info(f"    {len(all_groups_dov)} groups after join...")

        if verbose:
            logger.info("Extending to mask")
        all_groups_extdov = extend_groups_to_mask(
            all_groups_dov,
            break_mask,
            self.window_size,
            self.sr,
            self.timestep,
            toler=self.ext_mask_tol,
        )

        if verbose:
            logger.info("Trimming Silence")
        all_groups_ts = trim_silence(
            all_groups_extdov, self.pitch, self.window_size, self.sr, self.timestep
        )

        all_groups_final = [
            remove_group_duplicates(g, self.dupl_perc_overlap_intra)
            for g in all_groups_ts
        ]

        if verbose:
            logger.info("Convert sequences to pitch track timesteps")
        starts_seq, lengths_seq = convert_seqs_to_timestep(
            all_groups_final, self.window_size, self.sr, self.timestep
        )

        if verbose:
            logger.info("Applying exclusion functions")
        starts_seq_exc, lengths_seq_exc = remove_below_length(
            starts_seq, lengths_seq, self.timestep, self.min_pattern_length_seconds
        )

        starts = [p for p in starts_seq_exc if len(p) >= self.min_in_group]
        lengths = [p for p in lengths_seq_exc if len(p) >= self.min_in_group]

        starts_sec = [[x * self.timestep for x in p] for p in starts]
        lengths_sec = [[x * self.timestep for x in l] for l in lengths]

        return starts, lengths

    def display_matrix(self, X, title=None, title_size=9, figsize=(3, 3)):
        if not isinstance(X, np.ndarray):
            raise ValueError("X must be a 2d numpy array")

        fig, ax = plt.subplots(figsize=figsize)
        if title:
            plt.title(title, fontsize=title_size)
        ax.imshow(X, interpolation="nearest")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def display_all_matrices(self, title_size=9, figsize=(3, 3)):
        if not self.emphasized:
            raise Exception(
                "Please run self.emphasize_diagonals before attempting to extract segments."
            )

        self.display_matrix(
            self.X, "Self Similarity", title_size=title_size, figsize=figsize
        )
        self.display_matrix(
            self.X_conv, "Convolved", title_size=title_size, figsize=figsize
        )
        self.display_matrix(
            self.X_diag,
            f"Binarized (threshold={self.bin_thresh})",
            title_size=title_size,
            figsize=figsize,
        )
        if self.gauss_sigma:
            self.display_matrix(
                self.X_gauss,
                f"Diagonal gaussian blur (sigma={self.gauss_sigma})",
                title_size=title_size,
                figsize=figsize,
            )
            self.display_matrix(
                self.X_cont,
                f"Gaussian binarized (threshold={self.bin_thresh_segment})",
                title_size=title_size,
                figsize=figsize,
            )
        self.display_matrix(
            self.X_fill,
            f"Morphological opening (kernel size={self.etc_kernel_size})",
            title_size=title_size,
            figsize=figsize,
        )
        self.display_matrix(
            self.X_binop,
            f"Morphological closing (square dimension={self.binop_dim}",
            title_size=title_size,
            figsize=figsize,
        )
        self.display_matrix(
            self.X_proc, "Final Matrix", title_size=title_size, figsize=figsize
        )

    def print_steps(self):
        logger.info("Current Status")
        logger.info("--------------")
        logger.info(f"Input matrix of shape: {self.shape}")
        logger.info(f"Windows size: {self.window_size}")
        logger.info(f"Sampling rate of original audio: {self.sr}\n")

        logger.info("Segment extraction")
        logger.info("------------------")
        if self.emphasized:
            logger.info("Convolved matrix available at self.X_conv")
            logger.info(
                f"Binarized matrix available at self.X_bin (threshold={self.bin_thresh})"
            )
            if self.gauss_sigma:
                logger.info(
                    f"Gaussian smoothed matrix available at self.X_gauss (sigma={self.gauss_sigma})"
                )
            else:
                logger.info(f"No gaussian smoothing was applied")
            logger.info(
                f"Morphologically closed matrix available at self.X_fill (kernel size={self.etc_kernel_size})"
            )
            logger.info(
                f"Morphologically opened matrix available at self.X_binop (square dimension of binary opening structure={self.binop_dim})"
            )
            logger.info(
                "Final matrix after all steps applied, available at self.X_proc"
            )
        else:
            logger.info(
                "No processes have been applied to the input matrix (see self.emphasize_diagonals)"
            )

        if self.extracted:
            logger.info("")
        else:
            logger.info("No segments have been extracted (see self.extract_segments)")

    def cache_paths(self):
        return {
            "convolved": self.conv_path,
            "extracted_segments": self.seg_path,
            "extended_segments": self.seg_ext_path,
            "joined_segments": self.seg_join_path,
        }

    def clear_cache(self):
        shutil.rmtree(self.cache_dir)

    def __repr__(self):
        return f"segmentExtractor(X={self.shape}, window_size={self.window_size}, sr={self.sr}, cache_dir={self.cache_dir})"
