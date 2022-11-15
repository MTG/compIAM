from compiam.melody.pattern.sancara_search.extraction.img import (
    remove_diagonal, convolve_array_tile, binarize, diagonal_gaussian, 
    apply_bin_op, make_symmetric, edges_to_contours)
from compiam.melody.pattern.sancara_search.extraction.segments import (
    extract_segments_new, break_all_segments, remove_short, extend_segments, join_all_segments, 
    extend_groups_to_mask, group_segments, group_overlapping, group_by_distance)
from compiam.melody.pattern.sancara_search.extraction.sequence import (
    convert_seqs_to_timestep, remove_below_length)
from compiam.melody.pattern.sancara_search.extraction.evaluation import get_coverage

from compiam.melody.pattern.sancara_search.complex_auto.util import to_numpy
from scipy.spatial.distance import pdist, squareform

def self_similarity(features, exclusion_mask=None, timestep=None, hop_length=None, sr=44100):
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
    if exclusion_mask is not None:
        assert(all([timestep is not None, hop_length is not None, sr is not None])), \
            "To use exclusion mask, <timestep>, <hop_length> and <sr> must also be passed"

    # Deal with masking if any
    if exclusion_mask:
        features_mask = convert_mask(features, exclusion_mask, timestep, hop_length, sr)
        orig_sparse_lookup, sparse_orig_lookup, boundaries_orig, boundaries_sparse = get_conversion_mappings(features_mask)
    else:
        orig_sparse_lookup = None
        sparse_orig_lookup = None
        boundaries_orig = None
        boundaries_sparse = None
        
    # Indices we want to keep
    good_ix = np.where(mask==0)[0]

    # Compute self similarity
    sparse_features = features[good_ix]
    matrix_orig = create_ss_matrix(sparse_features)

    # Normalise self similarity matrix
    matrix_norm = normalise_self_sim(matrix)

    if exclusion_mask:
        return matrix_norm, orig_sparse_lookup, sparse_orig_lookup, boundaries_orig, boundaries_sparse
    else:
        return matrix_norm


def convert_mask(arr, exclusion_mask, timestep, hop_length, sr):
    """
    Get mask of excluded regions in the same dimension as array, <arr>

    :param arr: array corresponding to features extracted from audio
    :type arr: np.ndarray
    :param exclusion_mask: Mask indicating whether element should be excluded (different dimensions to <arr>)
    :type exclusion_mask: np.ndarray
    :param timestep: time in seconds between each element in <exclusion_mask>
    :type timestep: float
    :param hop_length: how many frames of audio correspond to each element in <arr>
    :type hop_length: int
    :param sr: sampling rate of audio from which <arr> was computed
    :type sr: int

    :returns: array of mask values equal in length to one dimension of <arr> - 0/1 is masked?
    :rtype: np.ndarray
    """
    # get mask of silent and stable regions
    mask = []
    for i in range(arr.shape[0]):
        # what is the time at this element of arr?
        t = (i+1)*hop_length/sr
        # find index in mask
        ix = round(t/timestep)
        # append mask value for this point
        mask.append(mask[ix])
    mask = np.array(mask)
    return mask


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
    good_ix = np.where(mask==0)[0]

    # Mapping between old and new indices
    orig_sparse_lookup = {g:s for s,g in enumerate(good_ix)}
    sparse_orig_lookup = {s:g for g,s in orig_sparse_lookup.items()}

    # Indices corresponding to boundaries 
    # between wanted and unwanted regions
    # in original array
    boundaries_orig = []
    for i in range(1, len(mask)):
        curr = mask[i]
        prev = mask[i-1]
        if curr==0 and prev==1:
            boundaries_orig.append(i)
        elif curr==1 and prev==0:
            boundaries_orig.append(i-1)

    # Boundaries corresponding to newly joined
    # regions in sparse array
    boundaries_sparse = np.array([orig_sparse_lookup[i] for i in boundaries_orig])

    # Boundaries contain two consecutive boundaries for each gap
    # but not if the excluded region leads to the end of the track
    red_boundaries_sparse = []
    boundaries_mask = [0]*len(boundaries_sparse)
    for i in range(len(boundaries_sparse)):
        if i==0:
            red_boundaries_sparse.append(boundaries_sparse[i])
            boundaries_mask[i]=1
        if boundaries_mask[i]==1:
            continue
        curr = boundaries_sparse[i]
        prev = boundaries_sparse[i-1]
        if curr-prev == 1:
            red_boundaries_sparse.append(prev)
            boundaries_mask[i]=1
            boundaries_mask[i-1]=1
        else:
            red_boundaries_sparse.append(curr)
            boundaries_mask[i]=1
    boundaries_sparse = np.array(sorted(list(set(red_boundaries_sparse))))

    return orig_sparse_lookup, sparse_orig_lookup, boundaries_orig, boundaries_sparse


def create_ss_matrix(feats, mode='cosine'):
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
    matrix = squareform(pdist(np.vstack(to_numpy(feats)),
                              metric=mode))
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
    diag_mask = (diag_mask - np.diag(np.ones(matrix.shape[0]))).astype(np.bool)

    mat_min = np.min(matrix[diag_mask])
    mat_max = np.max(matrix[diag_mask])

    # not sure if needed for now. TODO: revisit
    #matrix -= matrix.min()
    #matrix /= (matrix.max() + 1e-8)

    #for b in boundaries_sparse:
    #    matrix[:,b] = 1
    #    matrix[b,:] = 1

    matrix[~diag_mask] = 0

    return matrix





