import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.signal import convolve2d

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
    em = not (exclusion_mask is None)
    if em:
        assert(all([not timestep is None, not hop_length is None, not sr is None])), \
            "To use exclusion mask, <timestep>, <hop_length> and <sr> must also be passed"

    # Deal with masking if any
    if em:
        features_mask = convert_mask(features, exclusion_mask, timestep, hop_length, sr)
        orig_sparse_lookup, sparse_orig_lookup, boundaries_orig, boundaries_sparse = get_conversion_mappings(features_mask)
    else:
        orig_sparse_lookup = None
        sparse_orig_lookup = None
        boundaries_orig = None
        boundaries_sparse = None
        
    # Indices we want to keep
    good_ix = np.where(features_mask==0)[0]

    # Compute self similarity
    sparse_features = features[good_ix]
    matrix = create_ss_matrix(sparse_features)

    # Normalise self similarity matrix
    matrix_norm = normalise_self_sim(matrix)

    if em:
        return matrix_norm, orig_sparse_lookup, sparse_orig_lookup, boundaries_orig, boundaries_sparse
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
        t = (i+1)*hop_length/sr
        # find index in mask
        ix = round(t/timestep)
        # append mask value for this point
        try:
            new_mask.append(mask[ix])
        except:
            import ipdb; ipdb.set_trace()
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
    matrix = squareform(pdist(np.vstack(feats.detach().numpy()),
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

    matrix -= matrix.min()
    matrix /= (matrix.max() + 1e-8)

    matrix[~diag_mask] = 0

    return matrix

def extract_segments(X, image_report=False, cache_dir=None):
    """
    From self similarity matrix, <X>. Return list of segments,
    each corresponding to two regions of the input axis.
    
    :param X: normalised self similarity matrix
    :type X: np.ndarry

    :returns: list of segments in the form [((x0,y0),(x1,y1)),..]
    :rtype: list
    """

    if image_report:
        report_fns = get_report_paths(image_report)

    if cache_dir:
        cach = os.path.join(cache_dir, '{0}','')

    save_matrix(X, report_fns['sim'])

    ##############
    ## Pipeline ##
    ##############
    print('Convolving similarity matrix')
    
    # Hash all parameters used before segment finding to hash results later
    
    conv_path = get_param_hash_filepath((s1, s2, conv_filter_str))
    X_conv = run_or_cache(convolve_array_tile, [X, conv_filter], cach.format('convolve'))
    #X_conv = convolve_array_tile(X, cfilter=conv_filter)

    if save_imgs:
        skimage.io.imsave(conv_filename, X_conv)

    print('Binarizing convolved array')
    X_bin = binarize(X_conv, bin_thresh, filename=bin_filename)
    #X_bin = binarize(X_conv, 0.05, filename=bin_filename)

    print('Removing diagonal')
    X_diag = remove_diagonal(X_bin)

    if save_imgs:
        skimage.io.imsave(diag_filename, X_diag)

    if gauss_sigma:
        print('Applying diagonal gaussian filter')
        diagonal_gaussian(X_bin, gauss_sigma, filename=gauss_filename)

        print('Binarize gaussian blurred similarity matrix')
        binarize(X_gauss, cont_thresh, filename=cont_filename)
    else:
        X_gauss = X_diag
        X_cont  = X_gauss

    print('Ensuring symmetry between upper and lower triangle in array')
    X_sym = make_symmetric(X_cont)

    print('Identifying and isolating regions between edges')
    X_fill = edges_to_contours(X_sym, etc_kernel_size)

    if save_imgs:
        skimage.io.imsave(close_filename, X_fill)

    print('Cleaning isolated non-directional regions using morphological opening')
    X_binop = apply_bin_op(X_fill, binop_dim)

    print('Ensuring symmetry between upper and lower triangle in array')
    X_binop = make_symmetric(X_binop)

    if save_imgs:
        skimage.io.imsave(binop_filename, X_binop)
    
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim
    ))

    segment_path = os.path.join(out_dir, f'segments/{seg_hash}.pkl')
    all_segments = run_or_cache(extract_segments_new, [X_binop], segment_path)

    print('Extending Segments')
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim, perc_tail, bin_thresh_segment
    ))

    segment_path = os.path.join(out_dir, f'segments_extended/{seg_hash}.pkl')
    all_segments_extended = run_or_cache(extend_segments, [all_segments, X_sym, X_conv, perc_tail, bin_thresh_segment], segment_path)

    print(f'    {len(all_segments_extended)} extended segments...')

    from exploration.segments import line_through_points
    
    all_segments_extended_reduced = remove_short(all_segments_extended, 1)

    print('Converting sparse segment indices to original')
    boundaries_sparse = [x for x in boundaries_sparse if x != 0]
    all_segments_scaled_x = []
    for seg in all_segments_extended_reduced:
        ((x0, y0), (x1, y1)) = seg
        get_x, get_y = line_through_points(x0, y0, x1, y1)
        
        boundaries_in_x = sorted([i for i in boundaries_sparse if i >= x0 and i <= x1])
        current_x0 = x0
        if boundaries_in_x:
            for b in boundaries_in_x:
                x0_ = current_x0
                x1_ = b-1
                y0_ = int(get_y(x0_))
                y1_ = int(get_y(x1_))
                all_segments_scaled_x.append(((x0_, y0_), (x1_, y1_)))
                current_x0 = b+1

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
                y1_ = b-1
                x0_ = int(get_x(y0_))
                x1_ = int(get_x(y1_))
                all_segments_scaled.append(((x0_, y0_), (x1_, y1_)))
                current_y0 = b+1
            
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

    de = 0 if s1 is None else s1
        
    for i, seg in enumerate(all_segments_scaled_reduced):
        ((x0, y0), (x1, y1)) = seg
        while (x0 in boundaries_sparse) or (x1 in boundaries_sparse) or (y0 in boundaries_sparse) or (y1 in boundaries_sparse):
            if x0 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                x0 = x0+1
                x1 = x1
                y0 = round(get_y(x0))
                y1 = round(get_y(x1))

            if x1 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                x0 = x0
                x1 = x1-1
                y0 = round(get_y(x0))
                y1 = round(get_y(x1))

            if y0 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                y0 = y0+1
                y1 = y1
                x0 = round(get_x(y0))
                x1 = round(get_x(y1))

            if y1 in boundaries_sparse:
                get_x, get_y = line_through_points(x0, y0, x1, y1)
                y0 = y0
                y1 = y1-1
                x0 = round(get_x(y0))
                x1 = round(get_x(y1))

        x0_ = sparse_orig_lookup[x0+de]
        y0_ = sparse_orig_lookup[y0+de]
        x1_ = sparse_orig_lookup[x1+de]
        y1_ = sparse_orig_lookup[y1+de]
        all_segments_converted.append(((x0_, y0_), (x1_, y1_)))
    

    #   [(i,(get_grad(x),get_grad(y))) for i,(x,y) in enumerate(zip(all_segments_scaled_reduced, all_segments_converted)) if get_grad(x) != get_grad(y)]

    #   get_grad = lambda y: (y[1][1]-y[0][1])/(y[1][0]-y[0][0])
    #   def check_seg(seg,boundaries_sparse):
    #       ((x0, y0), (x1, y1)) = seg
    #       return any([[x for x in boundaries_sparse if x0 <= x <= x1], [y for y in boundaries_sparse if y0 <= y <= y1]])

    print('Joining segments that are sufficiently close')
    # Hash all parameters used before segment finding to hash results later
    seg_hash = str((
        s1, s2, conv_filter_str, bin_thresh, gauss_sigma,
        cont_thresh, etc_kernel_size, binop_dim, perc_tail, 
        bin_thresh_segment, min_diff_trav_seq
    ))

    segment_path = os.path.join(out_dir, f'segments_joined/{seg_hash}.pkl')

    all_segments_joined = run_or_cache(join_all_segments, [all_segments_converted, min_diff_trav_seq], segment_path)
    print(f'    {len(all_segments_joined)} joined segments...')

    print('Breaking segments with silent/stable regions')
    # Format - [[(x,y), (x1,y1)],...]
    all_broken_segments = break_all_segments(all_segments_joined, silence_mask, cqt_window, sr, timestep)
    all_broken_segments = break_all_segments(all_broken_segments, stable_mask, cqt_window, sr, timestep)
    print(f'    {len(all_broken_segments)} broken segments...')

    #[(i,((x0,y0), (x1,y1))) for i,((x0,y0), (x1,y1)) in enumerate(all_segments) if x1-x0>10000]
    print('Reducing Segments')
    all_segments_reduced = remove_short(all_broken_segments, min_length_cqt)
    print(f'    {len(all_segments_reduced)} segments above minimum length of {min_pattern_length_seconds}s...')





