from compiam.melody.pattern.sancara_search.extraction.img import (
    remove_diagonal, convolve_array_tile, binarize, diagonal_gaussian, 
    apply_bin_op, make_symmetric, edges_to_contours)
from compiam.melody.pattern.sancara_search.extraction.segments import (
    extract_segments_new, break_all_segments, remove_short, extend_segments, join_all_segments, 
    extend_groups_to_mask, group_segments, group_overlapping, group_by_distance)
from compiam.melody.pattern.sancara_search.extraction.sequence import (
    convert_seqs_to_timestep, remove_below_length)
from compiam.melody.pattern.sancara_search.extraction.evaluation import get_coverage

def run_pipeline(X, 
    conv_filter, bin_thresh, gauss_sigma, cont_thresh, etc_kernel_size, binop_dim, perc_tail, bin_thresh_segment,
    min_diff_trav_seq, silence_mask, cqt_window, sr, timestep, min_length_cqt, match_tol, extend_tol, dupl_perc_overlap, 
    n_dtw, thresh_dtw, thresh_cos, min_pattern_length_seconds, min_in_group):

    print('Convolving similarity matrix')
    X_conv = convolve_array_tile(X, cfilter=conv_filter)

    print('Binarizing convolved array')
    X_bin = binarize(X_conv, bin_thresh)

    print('Removing diagonal')
    X_diag = remove_diagonal(X_bin)

    if gauss_sigma:
        print('Applying diagonal gaussian filter')
        diagonal_gaussian(X_bin, gauss_sigma)

        print('Binarize gaussian blurred similarity matrix')
        binarize(X_gauss, cont_thresh)
    else:
        X_gauss = X_diag
        X_cont = X_gauss

    print('Ensuring symmetry between upper and lower triangle in array')
    X_sym = make_symmetric(X_cont)

    print('Identifying and isolating regions between edges')
    X_fill = edges_to_contours(X_sym, etc_kernel_size)

    print('Cleaning isolated non-directional regions using morphological opening')
    X_binop = apply_bin_op(X_fill, binop_dim)

    print('Ensuring symmetry between upper and lower triangle in array')
    X_binop = make_symmetric(X_binop)

    ## Join segments that are sufficiently close
    print('Extracting segments using flood fill and centroid')
    all_segments = extract_segments_new(X_binop)

    print('Extending Segments')
    all_segments_extended = extend_segments(all_segments, X_sym, X_conv, perc_tail, bin_thresh_segment)
    print(f'    {len(all_segments_extended)} extended segments...')

    print('Joining segments that are sufficiently close')
    all_segments_joined = join_all_segments(all_segments_extended, min_diff_trav_seq)
    print(f'    {len(all_segments_joined)} joined segments...')

    print('Breaking segments with silent/stable regions')
    # Format - [[(x,y), (x1,y1)],...]
    all_broken_segments = break_all_segments(all_segments_joined, silence_mask, cqt_window, sr, timestep)
    all_broken_segments = break_all_segments(all_broken_segments, stable_mask, cqt_window, sr, timestep)
    print(f'    {len(all_broken_segments)} broken segments...')

    print('Reducing Segments')
    all_segments_reduced = remove_short(all_broken_segments, min_length_cqt)
    print(f'    {len(all_segments_reduced)} segments above minimum length of {min_pattern_length_seconds}s...')

    print('Identifying Segment Groups')
    all_groups = group_segments(all_segments_reduced, min_length_cqt, match_tol, silence_and_stable_mask, cqt_window, timestep, sr)
    print(f'    {len(all_groups)} segment groups found...')

    print('Extending segments to silence')
    silence_and_stable_mask_2 = np.array([1 if any([i==2,j==2]) else 0 for i,j in zip(silence_mask, stable_mask)])
    all_groups_ext = extend_groups_to_mask(all_groups, silence_and_stable_mask_2, toler=extend_tol)

    print('Joining Groups of overlapping Sequences')
    all_groups_over = group_overlapping(all_groups_ext, dupl_perc_overlap)
    print(f'    {len(all_groups_over)} groups after join...')

    print('Grouping further using distance measures')
    all_groups_dtw = group_by_distance(all_groups_over, pitch, n_dtw, thresh_dtw, thresh_cos, cqt_window, sr, timestep)
    print(f'    {len(all_groups_dtw)} groups after join...')

    print('Convert sequences to pitch track timesteps')
    starts_seq, lengths_seq = convert_seqs_to_timestep(all_groups_dtw, cqt_window, sr, timestep)

    print('Applying exclusion functions')
    starts_seq_exc,  lengths_seq_exc = remove_below_length(starts_seq, lengths_seq, timestep, min_pattern_length_seconds)

    starts_seq_exc = [p for p in starts_seq_exc if len(p)>min_in_group]
    lengths_seq_exc = [p for p in lengths_seq_exc if len(p)>min_in_group]

    print('Extend all segments to stable or silence')
    starts_seq_ext, lengths_seq_ext = starts_seq_exc, lengths_seq_exc

    starts_sec_ext = [[x*timestep for x in p] for p in starts_seq_ext]
    lengths_sec_ext = [[x*timestep for x in l] for l in lengths_seq_ext]

    starts_seq_ext = [[int(x/timestep) for x in p] for p in starts_sec_ext]
    lengths_seq_ext = [[int(x/timestep) for x in l] for l in lengths_sec_ext]

    print('')
    n_patterns = sum([len(x) for x in starts_seq_ext])
    coverage = get_coverage(pitch, starts_seq_ext, lengths_seq_ext)
    print(f'Number of Patterns: {n_patterns}')
    print(f'Number of Groups: {len(starts_sec_ext)}')
    print(f'Coverage: {round(coverage,2)}')

    return starts_seq_ext, lengths_seq_ext