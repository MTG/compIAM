import itertools
from itertools import groupby
import random

random.seed(42)

import fastdtw
import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq

from scipy.ndimage import label, generate_binary_structure
from scipy.spatial.distance import cosine
import skimage.io
from sklearn.cluster import DBSCAN

from operator import itemgetter

import tqdm

def get_extremes(angle, dist, l):
    # Make a line with "num" points...

    # y = (dist - x*np.cos(angle))/np.sin(angle)
    # x = (dist - y*np.sin(angle))/np.cos(angle)

    x0 = 0
    y0 = int((dist - x0*np.cos(angle))/np.sin(angle))

    if (y0 < 0 or y0 > l):
        y0 = 0
        x0 = int((dist - y0*np.sin(angle))/np.cos(angle))

    x1 = l
    y1 = int((dist - x1*np.cos(angle))/np.sin(angle))

    if (y1 < 0 or y1 > l):
        y1 = l
        x1 = int((dist - y1*np.sin(angle))/np.cos(angle))

    return y0, x0, y1, x1


def extend_segment(segment, max_l, padding):
    """
    segment: segment start and end indices [x0,x1]
    max_l: maximum indice possible
    padding: percentage to extend
    """
    x0 = segment[0]
    x1 = segment[1]
    length = x1-x0
    ext = int(padding*length)
    return [max([0,x0-ext]), min([max_l,x1+ext])]


def get_indices_of_line(l, angle, dist):
    """
    Return indices in square matrix of <l>x<l> for Hough line defined by <angle> and <dist> 
    """ 
    # Get start and end points of line to traverse from angle and dist
    x0, y0, x1, y1 = get_extremes(angle, dist, l)

    # To ensure no lines are defined outside the grid (should not be passed to func really)
    if any([y1>l, x1>l, x0>l, y0>l, y1<0, x1<0, x0<0, y0<0]):
        return None, None

    # Length of line to traverse
    length = int(np.hypot(x1-x0, y1-y0))
    x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)

    # x and y indices corresponding to line
    x = x.astype(int)
    y = y.astype(int)

    return x, y


def get_label_indices(X, structure_size=2):
    s = generate_binary_structure(structure_size, structure_size)
    labels, numL = label(X, structure=s)
    label_indices = [(labels == i).nonzero() for i in range(1, numL+1)]
    return label_indices


def line_through_points(x0, y0, x1, y1):
    """
    return function to convert x->y and for y->x
    for straight line that passes through x0,y0 and x1,y1
    """
    centroids = [(x0,y0), (x1, y1)]
    x_coords, y_coords = zip(*centroids)
    
    # gradient and intercecpt of line passing through centroids
    A = vstack([x_coords, ones(len(x_coords))]).T
    m, c = lstsq(A, y_coords, rcond=None)[0]

    # functions for converting between
    # x and y on that line
    get_y = lambda xin: m*xin + c
    get_x = lambda yin: (yin - c)/m

    return get_x, get_y


def extract_segments_new(X):
    # size of square array
    n1, n2 = X.shape

    # Goal: for each thick diagonal extract one segment 
    # corresponding to pattern for that diagonal
    label_indices = get_label_indices(X)

    all_segments = []
    for ix, (x,y) in enumerate(label_indices):
        # (x,y) indices of points that define this diagonal.
        # These correspond to an area more than 1 element thick,
        # the objective is to identify a single path through
        # this area to nominate as the candidate underlying segment/pattern
        points = list(zip(x,y))

        # centroid of entire diagonal
        c_x, c_y = (int(sum(x) / len(points)), int(sum(y) / len(points)))

        # split into top left quadrant (tlq) and bottom right quadrant (brq) 
        #   (top right and bottom left quadrant are empty in diagonal bounding box)
        tlq = [(x,y) for x,y in points if x <= c_x and y <= c_y]
        brq = [(x,y) for x,y in points if x > c_x and y > c_y]

        tlq_x = [i[0] for i in tlq]
        tlq_y = [i[1] for i in tlq]

        brq_x = [i[0] for i in brq]
        brq_y = [i[1] for i in brq]

        if len(tlq) == 0 or len(brq) == 0:
            continue

        # Compute the centroid for each of the two quarters
        tlq_centroid = (int(sum(tlq_x) / len(tlq)), int(sum(tlq_y) / len(tlq)))
        brq_centroid = (int(sum(brq_x) / len(brq)), int(sum(brq_y) / len(brq)))

        # Get x and y limits of bounding box of entire area
        x_sorted = sorted(points, key=lambda y: y[0])
        y_sorted = sorted(points, key=lambda y: y[1])
        
        north_y = y_sorted[0][1] # line across the top
        south_y = y_sorted[-1][1] # line across the bottom
        west_x  = x_sorted[0][0] # line across left side
        east_x  = x_sorted[-1][0] # line across right side

        # functions for converting between
        # x and y on that line
        get_x, get_y = line_through_points(
            tlq_centroid[0], tlq_centroid[1], brq_centroid[0], brq_centroid[1])

        # does the line intersect the roof or sides of the bounding box?
        does_intersect_roof = get_y(west_x) > north_y

        if does_intersect_roof:
            y0 = north_y
            x0 = get_x(y0)

            y1 = south_y
            x1 = get_x(y1)
        else:
            x0 = west_x
            y0 = get_y(x0)

            x1 = east_x
            y1 = get_y(x1)
        
        # int() always rounds down
        roundit = lambda yin: int(round(yin))

        # Points are computed using a line learnt
        # using least squares, there is a small chance that 
        # this results in one of the coordinates being slightly
        # over the limits of the array, the rounding that occurs 
        # when converting to int may make this +/- 1 outside of array
        # limits
        if roundit(x0) < 0:
            x0 = 0
            y0 = roundit(get_y(0))
        if roundit(y0) < 0:
            y0 = 0
            x0 = roundit(get_x(0))
        if roundit(x1) >= n1:
            x1 = n1-1
            y1 = roundit(get_y(x1))
        if roundit(y1) >= n2:
            y1 = n2-1
            x1 = roundit(get_x(y1))
        
        if not any([roundit(x1) < roundit(x0), roundit(y1) < roundit(y1)]):
            all_segments.append([(roundit(x0), roundit(y0)), (roundit(x1), roundit(y1))])

    return all_segments


def extract_segments(matrix, angle, dist, min_diff, cqt_window, sr, padding=None):
    """
    Extract start and end coordinates of non-zero elements along hough line defined
    by <angle> and <dist>. If <padding>, extend length of each segment by <padding>%
    along the line.
    """
    # traverse hough lines and identify non-zero segments 
    l = matrix.shape[0]-1

    x, y = get_indices_of_line(l, angle, dist)
    
    # line defined outside of grid
    if x is None:
        return []

    max_l = len(x)-1

    # Extract the values along the line
    zi = matrix[x, y]

    # Get index of non-zero elements along line
    non_zero = np.where(zi != 0)[0]
    if len(non_zero) == 0:
        return []

    # Identify segments of continuous non-zero along line
    segments = []
    this_segment = []
    for i in range(len(non_zero)):
        # First non zero must be a start point of segment
        if i == 0:
            this_segment.append(non_zero[i])
            continue

        # Number of elements along hypotenuse
        n_elems = non_zero[i] - non_zero[i-1]

        # Time corresponding to gap found between this silence and previous
        #   - n_elems is length of hypotonuse in cqt space
        #   - (assume equilateral) divide by sqrt(2) to get adjacent length (length of gap)
        #   - mulitply by cqt_window and divide by sample rate to get adjacent length in seconds
        T = (cqt_window * n_elems) / (sr * 2**0.5)

        # If gap is smaller than min_diff, ignore it
        if T <= min_diff:
            continue
        else:
            # consider gap the end of found segment and store
            this_segment.append(non_zero[i-1])
            segments.append(this_segment)
            this_segment = [non_zero[i]]

    this_segment.append(non_zero[-1])

    if padding:
        this_segment = extend_segment(this_segment, max_l, padding)

    segments.append(this_segment)

    all_segments = []
    for i1, i2 in segments:
        # (x start, y start), (x end, y end)
        all_segments.append([(x[i1], y[i1]), (x[i2], y[i2])])

    return all_segments


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)


def extend_segments(all_segments, X_in, X_conv, perc_tail, bin_thresh_segment):
    h,w = X_in.shape

    all_segments_extended = []
    for ((x0, y0), (x1, y1)) in all_segments:    
        dx = (x1-x0)
        dy = (y1-y0)

        length = (dx**2 + dy**2)**0.5
        grad = dy/dx

        # Angle line makes with x axis
        theta = np.arctan(grad)

        # Length of extra part of line
        extra = length * perc_tail

        # Get new start and end points of extended segment
        Dx = extra/np.sin(theta)
        Dy = extra/np.cos(theta) 

        X0 = int(x0 - Dx)
        Y0 = int(y0 - Dy)
        X1 = int(x1 + Dx)
        Y1 = int(y1 + Dy)

        # Coordinates of line connecting X0, Y0 and X1, Y1
        new_length = round(length + 2*extra)
        X, Y = np.linspace(X0, X1, new_length), np.linspace(Y0, Y1, new_length)
        X = [round(x) for x in X]
        Y = [round(y) for y in Y]
        filts = [(x,y) for x,y in zip(X,Y) if all([x>=0, y>=0, x<w, y<h])]
        X = [x[0] for x in filts]
        Y = [x[1] for x in filts]

        # the line can be cut short because of matrix boundaries
        clos0 = closest_node((x0,y0), list(zip(X,Y)))
        clos1 = closest_node((x1,y1), list(zip(X,Y)))

        new_seg = X_conv[X,Y]>bin_thresh_segment
        # original segment is always 1
        new_seg[clos0:clos1+1] = 1

        i0 = clos0
        i1 = clos1
        # go backwards through preceeding extension until there are no more
        # values that correspond to similarity above threshold
        for i,v in list(enumerate(new_seg))[:clos0][::-1]:
            if v == 0:
                i0 = i + 1
                break

        # go forwards through succeeding extension until there are no more
        # values that correspond to similarity above threshold
        accum = 0
        for i,v in list(enumerate(new_seg))[clos1:]:
            if v == 0:
                accum += 1
            if accum == 3: # roughly 0.1 seconds of zero values signifies break point
                i1 = i - 1
                break

        x0_new = X[i0]
        y0_new = Y[i0]

        x1_new = X[i1]
        y1_new = Y[i1]

        ext_segment = [(x0_new, y0_new), (x1_new, y1_new)]
        all_segments_extended.append(ext_segment)

    return all_segments_extended


def get_all_segments(X, peaks, min_diff_trav, min_length_cqt, cqt_window, sr):
    all_segments = []
    for _, angle, dist in zip(*peaks):
        segments = extract_segments(X, angle, dist, min_diff_trav, cqt_window, sr)

        # If either of the lengths are above minimum length, add to all segments
        for s in segments:
            x0 = s[0][0]
            y0 = s[0][1]
            x1 = s[1][0]
            y1 = s[1][1]

            l0 = x1-x0
            l1 = y1-y0

            # temp | to_add = []
            # temp | if max([l1, l0]) > min_length_cqt:
                # temp | to_add.append(s)

            all_segments.append(s)

        # temp | all_segments += to_add

    all_segments = sorted([sorted(x) for x in all_segments])

    return all_segments


#   length_change = 1
#   while length_change != 0:
#       l1 = len(all_segments)
#       all_segments = sorted([sorted(x, key=lambda y: (y[0], y[1])) for x in all_segments])
#       all_segments = [all_segments[i] for i in range(len(all_segments)) \
#                           if i == 0 or not \
#                           (same_seqs_marriage(
#                               all_segments[i][0][0], all_segments[i][0][1],
#                               all_segments[i-1][0][0], all_segments[i-1][0][1],
#                                thresh=same_seqs_thresh) and
#                            same_seqs_marriage(
#                               all_segments[i][1][0], all_segments[i][1][1],
#                               all_segments[i-1][1][0], all_segments[i-1][1][1],
#                                thresh=same_seqs_thresh))
#                       ]
#       l2 = len(all_segments)
#       length_change=l2-l1


def break_segment(segment_pair, mask, cqt_window, sr, timestep):
    
    # (x start, y start), (x end, y end)
    x_start = segment_pair[0][0]
    x_start_ts = round((x_start*cqt_window)/(sr*timestep))

    x_end = segment_pair[1][0]
    x_end_ts = round((x_end*cqt_window)/(sr*timestep))

    y_start = segment_pair[0][1]
    y_start_ts = round((y_start*cqt_window)/(sr*timestep))

    y_end = segment_pair[1][1]
    y_end_ts = round((y_end*cqt_window)/(sr*timestep))

    stab_x = mask[x_start_ts:x_end_ts]
    stab_y = mask[y_start_ts:y_end_ts]

    # If either sequence contains a masked region, divide
    if any([2 in stab_x, 2 in stab_y]):
        break_points_x = np.where(stab_x==2)[0]
        break_points_y = np.where(stab_y==2)[0]
        if len(break_points_y) > len(break_points_x):
            bpy_ = break_points_y
            # break points x should correspond to the same proportion through the sequence
            # as break points y, since they can be different lengths
            bpx_ = [round((b/len(stab_y))*len(stab_x)) for b in bpy_]
            
            # convert back to cqt_window granularity sequence
            bpx = [round(x*(sr*timestep)/cqt_window) for x in bpx_]
            bpy = [round(y*(sr*timestep)/cqt_window) for y in bpy_]
        else:
            bpx_ = break_points_x
            # break points y should correspond to the same proportion through the sequence
            # as break points x, since they can be different lengths
            bpy_ = [round((b/len(stab_x))*len(stab_y)) for b in bpx_]
            
            # convert back to cqt_window granularity sequence
            bpy = [round(x*(sr*timestep)/cqt_window) for x in bpy_]
            bpx = [round(x*(sr*timestep)/cqt_window) for x in bpx_]
    else:
        # nothing to be broken, return original segment
        return [[(x_start, y_start), (x_end, y_end)]]

    new_segments = []
    for i in range(len(bpx)):
        bx = bpx[i]
        by = bpy[i]

        if i == 0:
            new_segments.append([(x_start, y_start), (x_start+bx, y_start+by)])
        else:
            # break points from last iterations
            # we begin on these this time
            bx1 = bpx[i-1]
            by1 = bpy[i-1]

            new_segments.append([(x_start+bx1, y_start+by1), (x_start+bx, y_start+by)])

    new_segments.append([(x_start+bx, y_start+by), (x_end, y_end)])

    return new_segments


def break_all_segments(all_segments, mask, cqt_window, sr, timestep):
    all_broken_segments = []
    for segment_pair in all_segments:
        broken = break_segment(segment_pair, mask, cqt_window, sr, timestep)
        # if there is nothing to break, the 
        # original segment pair is returned
        all_broken_segments += broken
    return sorted([sorted(x) for x in all_broken_segments])


def get_overlap(x0, x1, y0, y1):
    
    p0_indices = set(range(x0, x1+1))
    p1_indices = set(range(y0, y1+1))
    
    inters = p1_indices.intersection(p0_indices)

    o1 = len(inters)/len(p0_indices)
    o2 = len(inters)/len(p1_indices)
    
    return o1, o2


def do_patterns_overlap(x0, x1, y0, y1, perc_overlap):
    
    o1, o2 = get_overlap(x0, x1, y0, y1)

    return o1>perc_overlap and o2>perc_overlap


def do_segments_overlap(seg1, seg2, perc_overlap=0.5):
    """
    The Hough transform allows for the same segment to be intersected
    twice by lines of slightly different angle. We want to take the 
    longest of these duplicates and discard the rest
    
    Two segments inputed
      ...each formatted - [(x0, y0), (x1, y1)]
    These patterns could be distinct regions or not
    """
    # Assign the longest of the two segments to 
    # segment L and the shortest to segment S
    (x0, y0), (x1, y1) = seg1
    (x2, y2), (x3, y3) = seg2

    len_seg1 = np.hypot(x1-x0, y1-y0)
    len_seg2 = np.hypot(x3-x2, y3-y2)

    if len_seg1 >= len_seg2:
        seg_L = seg1
        seg_S = seg2
    else:
        seg_L = seg2
        seg_S = seg1

    # Each segment corresponds to two patterns
    #   - [segment 1] p0: x0 -> x1
    #   - [segment 1] p1: y0 -> y1
    #   - [segment 2] p2: x2 -> x3
    #   - [segment 2] p3: y2 -> y3

    (lx0, ly0), (lx1, ly1) = seg_L
    (sx0, sy0), (sx1, sy1) = seg_S

    # The two segments correspond to the same pair of patterns 
    # if p2 is a subset of p0 AND p3 is a subset of p2
    # We consider "subset" to mean > <perc_overlap>% overlap in indices
    overlap1 = do_patterns_overlap(lx0, lx1, sx0, sx1, perc_overlap=perc_overlap)
    overlap2 = do_patterns_overlap(ly0, ly1, sy0, sy1, perc_overlap=perc_overlap)

    # Return True if overlap in both dimensions
    return overlap1 and overlap2


def reduce_duplicates(all_segments, perc_overlap=0.5):
    all_seg_copy = all_segments.copy()

    # Order by length to speed computation
    seg_length = lambda y: np.hypot(y[1][0]-y[0][0], y[1][1]-y[0][1])
    all_seg_copy = sorted(all_seg_copy, key=seg_length, reverse=True)

    skip_array = [0]*len(all_seg_copy)
    reduced_segments = []
    # Iterate through all patterns and remove duplicates
    for i, seg1 in enumerate(all_seg_copy):
        # If this segment has been grouped already, do not consider
        if skip_array[i] == 1:
            continue

        for j, seg2 in enumerate(all_seg_copy[i+1:], i+1):
            # True or False, do they overlap in x and y?
            overlap = do_segments_overlap(seg1, seg2, perc_overlap=perc_overlap)

            # If they overlap discard seg2 (since it is shorter)
            if overlap:
                # remove this pattern
                skip_array[j] = 1

        # append original pattern
        reduced_segments += [seg1]

    return reduced_segments


def remove_short(all_segments, min_length_cqt):
    long_segs = []
    for (x0, y0), (x1, y1) in all_segments:
        length1 = x1 - x0
        length2 = y1 - y0
        if all([length1>min_length_cqt, length2>min_length_cqt]):
            long_segs.append([(x0, y0), (x1, y1)])
    return long_segs


def same_seqs_marriage(x1, y1, x2, y2, thresh=4):
    return (abs(x1-x2) < thresh) and (abs(y1-y2) < thresh)


def get_length(x1,y1,x2,y2):
    return ((x2-x1)**2 + (y2-y1)**2)**0.5


#def remove_group_duplicates(group, perc_overlap, both=False):
#    if len(group)==1:
#        return group
#    group_sorted = sorted(group, key= lambda y: (y[1]-y[0]), reverse=True)

#    new_group = []
#    skip_array = [0]*len(group_sorted)
#    for i,(x0,x1) in enumerate(group_sorted):
#        if skip_array[i]:
#            continue 
#        for j,(y0,y1) in enumerate(group_sorted):
#            if skip_array[j] or i>=j:
#                continue
#            overlap = do_patterns_overlap(x0, x1, y0, y1, perc_overlap=perc_overlap)
#            
#            if overlap:
#                if both:
#                    new_group.append((min([x0,y0]),max([x1,y1])))
#                # skip j since it is shorter
#                skip_array[j] = 1
#                continue
#        
#        if not both:
#            new_group.append((x0,x1))
#        else:
#            if (not overlap):
#                new_group.append((x0,x1))

#        skip_array[i] = 1
#    return new_group

def remove_group_duplicates(group, perc_overlap):
    group_sorted = sorted(group, key= lambda y: (y[1]-y[0]), reverse=True)

    new_group = []
    skip_array = [0]*len(group_sorted)
    for i,(x0,x1) in enumerate(group_sorted):
        if skip_array[i]:
            continue 
        this_group = [(x0,x1)]
        for j,(y0,y1) in enumerate(group_sorted):
            if skip_array[j] or i>=j:
                continue
            if y0 < x0 and x1 < y1:
                this_group.append((y0,y1))
                # skip j since it is shorter
                skip_array[j] = 1
            if x0 < y0 and y1 < x1:
                this_group.append((x0,x1))
                # skip j since it is shorter
                skip_array[j] = 1
            elif do_patterns_overlap(x0, x1, y0, y1, perc_overlap=perc_overlap):
                this_group.append((y0,y1))
                # skip j since it is shorter
                skip_array[j] = 1
                continue
        winx0 = min([x[0] for x in this_group])
        winx1 = max([x[1] for x in this_group])
        new_group.append((winx0, winx1))
        skip_array[i] = 1

    return new_group


def get_longest(x0,x1,y0,y1):
    len_x = x1-x0
    len_y = y1-y0
    if len_x > len_y:
        return x0, x1
    else:
        return y0, y1


def is_good_segment(x0, y0, x1, y1, thresh, silence_and_stable_mask, cqt_window, timestep, sr):
    try:
        x0s = round(x0*cqt_window/(sr*timestep))
        x1s = round(x1*cqt_window/(sr*timestep))
        y0s = round(y0*cqt_window/(sr*timestep))
        y1s = round(y1*cqt_window/(sr*timestep))

        seq1_stab = silence_and_stable_mask[x0s:x1s]
        seq2_stab = silence_and_stable_mask[y0s:y1s]

        prop_stab1 = sum(seq1_stab!=0) / len(seq1_stab)
        prop_stab2 = sum(seq2_stab!=0) / len(seq2_stab)

        if not (prop_stab1 > thresh or prop_stab2 > thresh):
            return True
        else:
            return False
    except:
        return False


def matches_dict_to_groups(matches_dict):
    l = [list(set([k]+v)) for k,v in matches_dict.items()]
    out = []
    while len(l)>0:
        first, *rest = l
        first = set(first)

        lf = -1
        while len(first)>lf:
            lf = len(first)

            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2

        out.append(first)
        l = rest

    return [list(o) for o in out]


def check_groups_unique(all_groups):
    repl = True
    for i,ag in enumerate(all_groups):
        for j,ag1 in enumerate(all_groups):
            if i==j:
                continue
            if set(ag).intersection(set(ag1)):
                print(f"groups {i} and {j} intersect")
                repl = False
    return repl


def update_dict(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


def ensure_indices_in_dict(d, ni):
    cd = d.copy()
    mi = max(d.keys())
    nd = {i:[] for i in range(mi+1,ni)}
    cd.update(nd)
    return cd


def get_dist(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


def join_segments(segA, segB):
    ((Ax0,Ay0), (Ax1,Ay1)) = segA
    ((Bx0,By0), (Bx1,By1)) =  segB
    # which starts closer to origin?
    # use that ones start point and the others end point
    if get_dist((Ax0,Ay0), (0,0)) > get_dist((Bx0,By0), (0,0)):
        x0 = Bx0
        y0 = By0
        x1 = Ax1
        y1 = Ay1
    else:
        x0 = Ax0
        y0 = Ay0
        x1 = Bx1
        y1 = By1
    return ((x0,y0), (x1, y1))


def join_all_segments(all_segments, min_diff_trav_seq):
    group_join_dict = {k:[] for k in range(len(all_segments))}
    for i, ((Qx0, Qy0), (Qx1, Qy1)) in tqdm.tqdm(list(enumerate(all_segments))):
        for j, [(Rx0, Ry0), (Rx1, Ry1)] in enumerate(all_segments):
            if i == j:
                continue

            if abs(Rx0-Qx1) > min_diff_trav_seq:
                continue
            if abs(Ry0-Qy1) > min_diff_trav_seq:
                continue

            gradR = np.arctan((Ry1-Ry0)/(Rx1-Rx0))
            gradQ = np.arctan((Qy1-Qy0)/(Qx1-Qx0))

            same_grad = abs(gradR-gradQ) <= 0.05

            # if distance between start an end
            if (get_dist((Rx0, Ry0), (Qx1, Qy1)) < min_diff_trav_seq) and same_grad:
                update_dict(group_join_dict, i, j)
                update_dict(group_join_dict, j, i)
                continue
            # if distance between end and start
            elif (get_dist((Rx1, Ry1), (Qx0, Qy0)) < min_diff_trav_seq):
                update_dict(group_join_dict, i, j)
                update_dict(group_join_dict, j, i)
                continue

    all_prox_groups = matches_dict_to_groups(group_join_dict)
    to_skip = []
    all_segments_joined = []
    for group in all_prox_groups:
        to_skip += group
        seg = all_segments[group[0]]
        for g in group[1:]:
            seg2 = all_segments[g]
            seg = join_segments(seg, seg2)
        all_segments_joined.append(seg)
    all_segments_joined += [x for i,x in enumerate(all_segments) if i not in to_skip]
    return all_segments_joined


def seg_contains_silent(seg, raw_pitch, cqt_window, sr, timestep):
    mat_to_pitch = lambda y: int((y*cqt_window)/(sr*timestep))
    x0 = seg[0][0]
    y0 = seg[0][1]
    x1 = seg[1][0]
    y1 = seg[1][1]
    x0_ = mat_to_pitch(x0)
    y0_ = mat_to_pitch(y0)
    y1_ = mat_to_pitch(y1)
    x1_ = mat_to_pitch(x1)
    x = np.trim_zeros(raw_pitch[x0_:x1_])
    y = np.trim_zeros(raw_pitch[y0_:y1_])
    if 0 in x:
        return True
    elif 0 in y:
        return True
    else:
        return False


def learn_relationships_and_break_x(
    i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, 
    new_segments, contains_dict, is_subset_dict, shares_common, raw_pitch, cqt_window, sr, timestep):
    """
    # Types of matches for two sequences: 
    #       query (Q):(------) and returned (R):[-------]
    # 1. (------) [------] - no match
    #   - Do nothing
    # 2. (-----[-)-----] - insignificant overlap
    #   - Do nothing
    # 3. (-[------)-] - left not significant, overlap significant, right not significant
    #   - Group Q and R


    # Query is on the left: Qx0 < Rx0
    #################################
    # 4. (-[-------)--------] - left not significant, overlap significant, right significant
    #   - Cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R1 and Q to group
    #   - R2 and R1 marked as new segments
    # 5. (---------[------)-] - left significant, overlap significant, right not significant
    #   - Cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - Add Q2 and R to group
    #   - Q1 and Q2 marked as new segments
    # 6. (---------[------)-------] - left signeificant, overlap significant, right significant
    #   - cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add Q2 and R1 to group
    #   - Q1, Q2, R1 and R2 marked as new segments


    # Query is on the left: Rx0 < Qx0
    #################################
    # 7. [-(-------]--------) - left not significant, overlap significant, right significant
    #   - Cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - Add R and Q1 to group
    #   - Q1 and Q2 marked as new segments
    # 8. [---------(------]-) - left significant, overlap significant, right not significant
    #   - Cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R2 and Q to group
    #   - R1 and R2 marked as new segments
    # 9. [---------(------]-------) - left significant, overlap significant, right significant
    #   - cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R2 and Q1 to group
    #   - Q1, Q2, R1 and R2 marked as new segments

    """
    # functions that define line through query(Q) segment
    Qget_x, Qget_y = line_through_points(Qx0, Qy0, Qx1, Qy1)
    # get indices corresponding to query(Q)
    Q_indices = set(range(Qx0, Qx1+1))

    # functions that define line through returned(R) segment
    Rget_x, Rget_y = line_through_points(Rx0, Ry0, Rx1, Ry1)
    # get indices corresponding to query(Q)
    R_indices = set(range(Rx0, Rx1+1))


    # query on the left
    if Qx0 <= Rx0:
        # indices in common between query(Q) and returned(R)
        left_indices = Q_indices.difference(R_indices)
        overlap_indices = Q_indices.intersection(R_indices)
        right_indices = R_indices.difference(Q_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = all([not left_sig, overlap_sig, right_sig])
        type_5 = all([left_sig, overlap_sig, not right_sig])
        type_6 = all([left_sig, overlap_sig, right_sig])

        type_7 = False
        type_8 = False
        type_9 = False

    # query on the right
    else:
        # indices in common between query(Q) and returned(R)
        left_indices = R_indices.difference(Q_indices)
        overlap_indices = R_indices.intersection(Q_indices)
        right_indices = Q_indices.difference(R_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = False
        type_5 = False
        type_6 = False

        type_7 = all([not left_sig, overlap_sig, right_sig])
        type_8 = all([left_sig, overlap_sig, not right_sig])
        type_9 = all([left_sig, overlap_sig, right_sig])

    ###########################
    ### Create New Segments ###
    ###########################
    if type_3:
        x0 = round(min(overlap_indices))
        x1 = round(max(overlap_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)

    if type_4 or type_7:
        x0 = round(min(overlap_indices))
        x1 = round(max(overlap_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)
        
            # index of new segment
            Oi = len(new_segments) - 1
        else:
            Oi = None

        x0 = round(min(right_indices))
        x1 = round(max(right_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        right_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(right_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(right_seg)

            # index of new segment
            Ri = len(new_segments) - 1
        else:
            Ri = None

    if type_5 or type_8:
        x0 = round(min(overlap_indices))
        x1 = round(max(overlap_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)

            # index of new segment
            Oi = len(new_segments) - 1
        else:
            Oi = None

        x0 = round(min(left_indices))
        x1 = round(max(left_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        left_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(left_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(left_seg)

            # index of new segment
            Li = len(new_segments) - 1
        else:
            Li = None

    if type_6 or type_9:
        x0 = round(min(overlap_indices))
        x1 = round(max(overlap_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)

            # index of new segment
            Oi = len(new_segments) - 1
        else:
            Oi = None

        x0 = round(min(left_indices))
        x1 = round(max(left_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        left_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(left_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(left_seg)

            # index of new segment
            Li = len(new_segments) - 1
        else:
            Li = None

        x0 = round(min(right_indices))
        x1 = round(max(right_indices))
        y0 = round(Qget_y(x0))
        y1 = round(Qget_y(x1))
        right_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(right_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(right_seg)  

            # index of new segment
            Ri = len(new_segments) - 1
        else:
            Ri = None

    ############################
    ### Record Relationships ###
    ############################
    if type_4:
        update_dict(contains_dict, j, i)
        if Oi:
            if j:
                update_dict(contains_dict, j, Oi)
        if Ri:
            update_dict(contains_dict, j, Ri)

        update_dict(is_subset_dict, i, j)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Ri:
            update_dict(is_subset_dict, Ri, j)

    if type_5:
        update_dict(contains_dict, i, j)
        if Oi:
            update_dict(contains_dict, i, Oi)
        if Li:
            update_dict(contains_dict, i, Li)

        update_dict(is_subset_dict, j, i)
        if Oi:
            update_dict(is_subset_dict, Oi, i)
        if Li:
            update_dict(is_subset_dict, Li, i)

    if type_6:
        if Oi:
            update_dict(contains_dict, i, Oi)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Li:
            update_dict(contains_dict, i, Li)
        if Ri:
            update_dict(contains_dict, j, Ri)

        if Oi:
            update_dict(is_subset_dict, Oi, i)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Li:
            update_dict(is_subset_dict, Li, i)
        if Ri:
            update_dict(is_subset_dict, Ri, j)

        update_dict(shares_common, i, j)
        update_dict(shares_common, j, i)

    if type_7:
        update_dict(contains_dict, j, i)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Ri:
            update_dict(contains_dict, j, Ri)

        update_dict(is_subset_dict, i, j)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Ri:
            update_dict(is_subset_dict, Ri, j)

    if type_8:
        update_dict(contains_dict, j, j)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Li:
            update_dict(contains_dict, j, Li)

        update_dict(is_subset_dict, j, j)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Li:
            update_dict(is_subset_dict, Li, j)

    if type_9:
        if Oi:
            update_dict(contains_dict, i, Oi)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Li:
            update_dict(contains_dict, j, Li)
        if Ri:
            update_dict(contains_dict, i, Ri)

        if Oi:
            update_dict(is_subset_dict, Oi, i)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Li:
            update_dict(is_subset_dict, Li, j)
        if Ri:
            update_dict(is_subset_dict, Ri, i)

        update_dict(shares_common, i, j)
        update_dict(shares_common, j, i)

    return new_segments, shares_common, is_subset_dict, contains_dict


def learn_relationships_and_break_y(
    i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, 
    new_segments, contains_dict, is_subset_dict, shares_common, raw_pitch, cqt_window, sr, timestep):
    """
    # Types of matches for two sequences: 
    #       query (Q):(------) and returned (R):[-------]
    # 1. (------) [------] - no match
    #   - Do nothing
    # 2. (-----[-)-----] - insignificant overlap
    #   - Do nothing
    # 3. (-[------)-] - left not significant, overlap significant, right not significant
    #   - Group Q and R


    # Query is on the left: Qx0 < Rx0
    #################################
    # 4. (-[-------)--------] - left not significant, overlap significant, right significant
    #   - Cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R1 and Q to group
    #   - R2 and R1 marked as new segments
    # 5. (---------[------)-] - left significant, overlap significant, right not significant
    #   - Cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - Add Q2 and R to group
    #   - Q1 and Q2 marked as new segments
    # 6. (---------[------)-------] - left significant, overlap significant, right significant
    #   - cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add Q2 and R1 to group
    #   - Q1, Q2, R1 and R2 marked as new segments


    # Query is on the left: Rx0 < Qx0
    #################################
    # 7. [-(-------]--------) - left not significant, overlap significant, right significant
    #   - Cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - Add R and Q1 to group
    #   - Q1 and Q2 marked as new segments
    # 8. [---------(------]-) - left significant, overlap significant, right not significant
    #   - Cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R2 and Q to group
    #   - R1 and R2 marked as new segments
    # 9. [---------(------]-------) - left significant, overlap significant, right significant
    #   - cut Q to create Q1 and Q2 (where Q1+Q2 = Q)
    #   - cut R to create R1 and R2 (where R1+R2 = R)
    #   - Add R2 and Q1 to group
    #   - Q1, Q2, R1 and R2 marked as new segments

    """
    # functions that define line through query(Q) segment
    Qget_x, Qget_y = line_through_points(Qx0, Qy0, Qx1, Qy1)
    # get indices corresponding to query(Q)
    Q_indices = set(range(Qy0, Qy1+1))

    # functions that define line through returned(R) segment
    Rget_x, Rget_y = line_through_points(Rx0, Ry0, Rx1, Ry1)
    # get indices corresponding to query(Q)
    R_indices = set(range(Ry0, Ry1+1))


    # query on the left
    if Ry0 <= Qy0:
        # indices in common between query(Q) and returned(R)
        left_indices = Q_indices.difference(R_indices)
        overlap_indices = Q_indices.intersection(R_indices)
        right_indices = R_indices.difference(Q_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = all([not left_sig, overlap_sig, right_sig])
        type_5 = all([left_sig, overlap_sig, not right_sig])
        type_6 = all([left_sig, overlap_sig, right_sig])

        type_7 = False
        type_8 = False
        type_9 = False

    # query on the right
    else:
        # indices in common between query(Q) and returned(R)
        left_indices = R_indices.difference(Q_indices)
        overlap_indices = R_indices.intersection(Q_indices)
        right_indices = Q_indices.difference(R_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) >= min_length_cqt
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) >= min_length_cqt

        # which type of match (if any). 
        # See above for explanation
        type_1 = not overlap_indices
        type_2 = not overlap_sig and overlap_indices
        type_3 = all([overlap_sig, not left_sig, not right_sig])

        type_4 = False
        type_5 = False
        type_6 = False

        type_7 = all([not left_sig, overlap_sig, right_sig])
        type_8 = all([left_sig, overlap_sig, not right_sig])
        type_9 = all([left_sig, overlap_sig, right_sig])

    ###########################
    ### Create New Segments ###
    ###########################
    if type_3:
        y0 = round(min(overlap_indices))
        y1 = round(max(overlap_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)


    if type_4 or type_7:
        y0 = round(min(overlap_indices))
        y1 = round(max(overlap_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)
        
            # index of new segment
            Oi = len(new_segments) - 1
        else:
            Oi = None

        y0 = round(min(right_indices))
        y1 = round(max(right_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        right_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(right_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(right_seg)
            # index of new segment
            Ri = len(new_segments) - 1
        else:
            Ri = None

    if type_5 or type_8:
        y0 = round(min(overlap_indices))
        y1 = round(max(overlap_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)

            # index of new segment
            Oi = len(new_segments) - 1
        else:
            Oi = None

        y0 = round(min(left_indices))
        y1 = round(max(left_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        left_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(left_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(left_seg)

            # index of new segment
            Li = len(new_segments) - 1
        else:
            Li = None

    if type_6 or type_9:
        y0 = round(min(overlap_indices))
        y1 = round(max(overlap_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        overlap_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(overlap_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(overlap_seg)

            # index of new segment
            Oi = len(new_segments) - 1
        else:
            Oi = None

        y0 = round(min(left_indices))
        y1 = round(max(left_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        left_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(left_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(left_seg)

            # index of new segment
            Li = len(new_segments) - 1
        else:
            Li = None

        y0 = round(min(right_indices))
        y1 = round(max(right_indices))
        x0 = round(Qget_x(y0))
        x1 = round(Qget_x(y1))
        right_seg = ((x0,y0),(x1,y1))
        if not seg_contains_silent(right_seg, raw_pitch, cqt_window, sr, timestep):
            new_segments.append(right_seg)  

            # index of new segment
            Ri = len(new_segments) - 1
        else:
            Ri = None

    ############################
    ### Record Relationships ###
    ############################
    if type_4:
        update_dict(contains_dict, j, i)
        if Oi:
            if j:
                update_dict(contains_dict, j, Oi)
        if Ri:
            update_dict(contains_dict, j, Ri)

        update_dict(is_subset_dict, i, j)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Ri:
            update_dict(is_subset_dict, Ri, j)

    if type_5:
        update_dict(contains_dict, i, j)
        if Oi:
            update_dict(contains_dict, i, Oi)
        if Li:
            update_dict(contains_dict, i, Li)

        update_dict(is_subset_dict, j, i)
        if Oi:
            update_dict(is_subset_dict, Oi, i)
        if Li:
            update_dict(is_subset_dict, Li, i)

    if type_6:
        if Oi:
            update_dict(contains_dict, i, Oi)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Li:
            update_dict(contains_dict, i, Li)
        if Ri:
            update_dict(contains_dict, j, Ri)

        if Oi:
            update_dict(is_subset_dict, Oi, i)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Li:
            update_dict(is_subset_dict, Li, i)
        if Ri:
            update_dict(is_subset_dict, Ri, j)

        update_dict(shares_common, i, j)
        update_dict(shares_common, j, i)

    if type_7:
        update_dict(contains_dict, j, i)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Ri:
            update_dict(contains_dict, j, Ri)

        update_dict(is_subset_dict, i, j)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Ri:
            update_dict(is_subset_dict, Ri, j)

    if type_8:
        update_dict(contains_dict, j, j)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Li:
            update_dict(contains_dict, j, Li)

        update_dict(is_subset_dict, j, j)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Li:
            update_dict(is_subset_dict, Li, j)

    if type_9:
        if Oi:
            update_dict(contains_dict, i, Oi)
        if Oi:
            update_dict(contains_dict, j, Oi)
        if Li:
            update_dict(contains_dict, j, Li)
        if Ri:
            update_dict(contains_dict, i, Ri)

        if Oi:
            update_dict(is_subset_dict, Oi, i)
        if Oi:
            update_dict(is_subset_dict, Oi, j)
        if Li:
            update_dict(is_subset_dict, Li, j)
        if Ri:
            update_dict(is_subset_dict, Ri, i)

        update_dict(shares_common, i, j)
        update_dict(shares_common, j, i)

    return new_segments, shares_common, is_subset_dict, contains_dict


def learn_relationships(all_segments, min_length_cqt, raw_pitch, cqt_window, sr, timestep):
    contains_dict = {k:[] for k in range(len(all_segments))}
    is_subset_dict = {k:[] for k in range(len(all_segments))}
    shares_common = {k:[] for k in range(len(all_segments))}

    new_segments = []

    for i, ((Qx0, Qy0), (Qx1, Qy1)) in tqdm.tqdm(list(enumerate(all_segments))):
        for j, [(Rx0, Ry0), (Rx1, Ry1)] in enumerate(all_segments): 
            if (Qx0 <= Rx0 <= Qx1) or (Rx0 <= Qx0 <= Rx1):
                # horizontal pass
                res = learn_relationships_and_break_x(
                    i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, 
                    new_segments, contains_dict, is_subset_dict, shares_common, raw_pitch, cqt_window, sr, timestep)
                new_segments, shares_common, is_subset_dict, contains_dict = res

            if (Qy0 <= Ry0 <= Qy1) or (Ry0 <= Qy0 <= Ry1):
                # vertical pass (swap xs and ys)
                res = learn_relationships_and_break_y(
                    i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, 
                    new_segments, contains_dict, is_subset_dict, shares_common, raw_pitch, cqt_window, sr, timestep)
                new_segments, shares_common, is_subset_dict, contains_dict = res
    return new_segments, contains_dict, is_subset_dict, shares_common


def identify_matches_x(i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, tol, matches_dict):

    # get indices corresponding to query(Q)
    Q_indices = set(range(Qx0, Qx1+1))

    # get indices corresponding to returned(R)
    R_indices = set(range(Rx0, Rx1+1))

    # query on the left
    if Qx0 <= Rx0:
        # indices in common between query(Q) and returned(R)
        left_indices = Q_indices.difference(R_indices)
        overlap_indices = Q_indices.intersection(R_indices)
        right_indices = R_indices.difference(Q_indices)

    # query on the right
    else:
        # indices in common between query(Q) and returned(R)
        left_indices = R_indices.difference(Q_indices)
        overlap_indices = R_indices.intersection(Q_indices)
        right_indices = Q_indices.difference(R_indices)

    # which parts in the venn diagram
    # betweem Q and R are large enough to
    # be considered
    left_sig = len(left_indices) > tol
    overlap_sig = len(overlap_indices) >= min_length_cqt
    right_sig = len(right_indices) > tol

    # exact matches only, relationships are captured previously
    if all([overlap_sig, not left_sig, not right_sig]):
        update_dict(matches_dict, i, j)
        update_dict(matches_dict, j, i)

    return matches_dict


def identify_matches_y(i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, Ry1, min_length_cqt, tol, matches_dict):

    # get indices corresponding to query(Q)
    Q_indices = set(range(Qy0, Qy1+1))

    # get indices corresponding to returned(R)
    R_indices = set(range(Ry0, Ry1+1))

    # query on the left
    if Ry0 <= Qy0:
        # indices in common between query(Q) and returned(R)
        left_indices = Q_indices.difference(R_indices)
        overlap_indices = Q_indices.intersection(R_indices)
        right_indices = R_indices.difference(Q_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) > tol
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) > tol

    # query on the right
    else:
        # indices in common between query(Q) and returned(R)
        left_indices = R_indices.difference(Q_indices)
        overlap_indices = R_indices.intersection(Q_indices)
        right_indices = Q_indices.difference(R_indices)

        # which parts in the venn diagram
        # betweem Q and R are large enough to
        # be considered
        left_sig = len(left_indices) > tol
        overlap_sig = len(overlap_indices) >= min_length_cqt
        right_sig = len(right_indices) > tol

    # exact matches only, relationships are captured previously
    if all([overlap_sig, not left_sig, not right_sig]):
        update_dict(matches_dict, i, j)
        update_dict(matches_dict, j, i)

    return matches_dict


def get_matches_dict(new_segments, min_length_cqt, match_tol):
    matches_dict = {k:[] for k in range(len(new_segments))}
    for i, ((Qx0, Qy0), (Qx1, Qy1)) in tqdm.tqdm(list(enumerate(new_segments))):
        for j, [(Rx0, Ry0), (Rx1, Ry1)] in enumerate(new_segments):
            if (Qx0 <= Rx0 <= Qx1) or (Rx0 <= Qx0 <= Rx1):
                # horizontal pass
                matches_dict = identify_matches_x(
                    i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, 
                    Ry1, min_length_cqt, match_tol, matches_dict) 

            if (Qy0 <= Ry0 <= Qy1) or (Ry0 <= Qy0 <= Ry1):
                # vertical pass (swap xs and ys)
                matches_dict = identify_matches_y(
                    i, j, Qx0, Qy0, Qx1, Qy1, Rx0, Ry0, Rx1, 
                    Ry1, min_length_cqt, match_tol, matches_dict) 

    return matches_dict


def get_segment_grouping(new_segments, matches_dict, silence_and_stable_mask, cqt_window, timestep, sr):

    segment_ix_dict = {i:((x0,y0), (x1,y1)) for i,((x0,y0), (x1,y1)) in enumerate(new_segments) \
                        if is_good_segment(x0, y0, x1, y1, 0.6, silence_and_stable_mask, cqt_window, timestep, sr)}

    print('...finalising grouping\n')
    all_groups = matches_dict_to_groups(matches_dict)
    check_groups_unique(all_groups)

    all_groups = [[segment_ix_dict[i] for i in ag if i in segment_ix_dict] for ag in all_groups]
    all_groups  = [[((x0,x1),(y0,y1)) for ((x0,y0),(x1,y1)) in ag] for ag in all_groups]

    all_groups  = [sorted([x for y in ag for x in y]) for ag in all_groups]

    all_groups = [sorted(ag, key=lambda y:y[0]) for ag in all_groups]

    all_groups = sorted(all_groups, key=lambda y: -len(y))
    all_groups = [x for x in all_groups if len(x) > 0]
    #all_groups = [remove_group_duplicates(g, 0.01) for g in all_groups]

    return all_groups


def group_segments(all_segments, min_length_cqt, match_tol, silence_and_stable_mask, cqt_window, timestep, sr, raw_pitch):
    print('...learning relationship between (sub)segments\n')
    new_segments, contains_dict, is_subset_dict, shares_common = learn_relationships(all_segments, min_length_cqt, raw_pitch, cqt_window, sr, timestep)

    print('...identifying matches\n')
    matches_dict = get_matches_dict(new_segments, min_length_cqt, match_tol)
    
    print('...identifying matches\n')
    all_groups = get_segment_grouping(new_segments, matches_dict, silence_and_stable_mask, cqt_window, timestep, sr)

    return all_groups


def extend_groups_to_mask(all_groups, mask, cqt_window, sr, timestep, toler=0.25):
    mask_i = list(range(len(mask)))
    new_groups = []
    for group in all_groups:
        this_group = []
        for x1, x2 in group:
            s1 = round((x1*cqt_window)/(sr*timestep))
            s2 = round((x2*cqt_window)/(sr*timestep))

            tol_ts = round(toler/timestep)
            
            s1_ = s1 - tol_ts
            s2_ = s2 + tol_ts

            midpoint = s1_ + round((s2_ - s1_)/2)

            s1_mask   = list(mask[s1_:s1])
            s2_mask   = list(mask[s2:s2_])
            s1_mask_i = list(mask_i[s1_:s1])
            s2_mask_i = list(mask_i[s2:s2_])

            if 1 in s1_mask:
                ix = len(s1_mask) - s1_mask[::-1].index(1) - 1
                s1 = s1_mask_i[ix]

            if 1 in s2_mask:
                ix = s2_mask.index(1)
                s2 = s2_mask_i[ix]

            x1 = round((s1*sr*timestep)/cqt_window)
            x2 = round((s2*sr*timestep)/cqt_window)

            this_group.append((x1,x2))

        new_groups.append(this_group)

    return new_groups



def countLeadingZeros(x, flip=False):
    """ Count number of elements up to the first non-zero element, return that count """
    X = np.flip(x) if flip else x
    ctr = 0
    for k in X:
        if k == 0:
            ctr += 1
        else: #short circuit evaluation, we found a non-zero so return immediately
            return ctr
    return ctr #we get here in the case that x was all zeros


def countLeadingNonZeros(x, flip=False):
    """ Count number of elements up to the first non-zero element, return that count """
    X = np.flip(x) if flip else x
    ctr = 0
    for k in X:
        if k != 0:
            ctr += 1
        else: #short circuit evaluation, we found a non-zero so return immediately
            return ctr
    return ctr #we get here in the case that x was all zeros


def trim_silence(all_groups_ext, pitch, cqt_window, sr, timestep):
    new_groups = []
    for group in all_groups_ext:
        this_group = []
        for x1, x2 in group:
            s1 = int(np.floor((x1*cqt_window)/(sr*timestep)))
            s2 = int(np.ceil((x2*cqt_window)/(sr*timestep)))

            l = s2-s1

            pitch_track = pitch[s1:s2]
            
            leading_nonzeros = countLeadingNonZeros(pitch_track)
            trailing_nonzeros = countLeadingNonZeros(pitch_track, flip=True)            
            
            if leading_nonzeros/len(pitch_track) <= 0.10:
                s1 = s1+leading_nonzeros
            if trailing_nonzeros/len(pitch_track) <= 0.10:
                s2 = s2-trailing_nonzeros
            
            pitch_track = pitch[s1:s2]

            leading_zeros = countLeadingZeros(pitch_track)
            trailing_zeros = countLeadingZeros(pitch_track, flip=True)

            s1 = s1+leading_zeros
            s2 = s2-trailing_zeros
            
            x1 = round((s1*sr*timestep)/cqt_window)
            x2 = round((s2*sr*timestep)/cqt_window)

            this_group.append((x1,x2))

        new_groups.append(this_group)

    return new_groups    

def same_group(group1, group2, perc_overlap, group_len_var, num=2):
    av_lenmax1 = np.mean([x1-x0 for x0,x1 in group1])
    av_lenmin1 = np.mean([x1-x0 for x0,x1 in group1])
    av_lenmax2 = np.mean([x1-x0 for x0,x1 in group2])
    av_lenmin2 = np.mean([x1-x0 for x0,x1 in group2])

    if abs(1-av_lenmax1/av_lenmin2) > group_len_var or abs(1-av_lenmax2/av_lenmin1) > group_len_var: 
        return False
    for x0,x1 in group1:
        for y0,y1 in group2:
            #xlen = (x1-x0)
            #lim = 0.3*xlen
            #ylen = y1-y0
            #if (ylen > xlen-lim) and (ylen < xlen+lim):
            overlap = do_patterns_overlap(x0, x1, y0, y1, perc_overlap=perc_overlap)
            #else:
            #    overlap = False
            if overlap:
                num -= 1
            if num == 0:
                return True
    return False


def group_overlapping(all_groups, dupl_perc_overlap, group_len_var):
    ## Remove those that are identical
    group_match_dict = {k:[] for k in range(len(all_groups))}
    for i, ag1 in tqdm.tqdm(list(enumerate(all_groups))):
        for j, ag2 in enumerate(all_groups):
            if i >= j:
                continue
            
            # If already paired, do not compute distance
            if i in group_match_dict[j] or j in group_match_dict[i]:
                continue

            if same_group(ag1, ag2, dupl_perc_overlap, group_len_var, 1):
                update_dict(group_match_dict, i, j)
                update_dict(group_match_dict, j, i)

    all_groups_ix = matches_dict_to_groups(group_match_dict)
    all_groups_ix = [list(set(x)) for x in all_groups_ix]
    all_groups = [[x for i in group for x in all_groups[i]] for group in all_groups_ix]
    #all_groups = [remove_group_duplicates(g, 0.01) for g in all_groups]

    return all_groups


def group_by_distance(all_groups, pitch, n_dtw, thresh_dtw, thresh_cos, group_len_var, cqt_window, sr, timestep):
    to_seqs = lambda y: round((y*cqt_window)/(sr*timestep))
    ## Remove those that are identical
    group_match_dict = {k:[] for k in range(len(all_groups))}

    for i, ag1 in tqdm.tqdm(list(enumerate(all_groups))):
        for j, ag2 in enumerate(all_groups):
            if j >= i:
                continue

            # If already paired, do not compute distance
            if i in group_match_dict[j] or j in group_match_dict[i]:
                continue
            
            av_lenmax1 = np.mean([x1-x0 for x0,x1 in ag1])
            av_lenmin1 = np.mean([x1-x0 for x0,x1 in ag1])
            av_lenmax2 = np.mean([x1-x0 for x0,x1 in ag2])
            av_lenmin2 = np.mean([x1-x0 for x0,x1 in ag2])

            if abs(1-av_lenmax1/av_lenmin2) > group_len_var or abs(1-av_lenmax2/av_lenmin1) > group_len_var: 
                continue

            sample1 = random.sample(ag1, min(n_dtw,len(ag1)))
            sample2 = random.sample(ag2, min(n_dtw,len(ag2)))

            dtw_total = 0
            N = 0
            for (x0, x1), (y0, y1) in itertools.product(sample1, sample2):
                #xlen = (x1-x0)
                #lim = 0.3*xlen
                #ylen = y1-y0
                #if (ylen > xlen-lim) and (ylen < xlen+lim):
                seq1 = np.trim_zeros(pitch[to_seqs(x0): to_seqs(x1)])
                seq2 = np.trim_zeros(pitch[to_seqs(y0): to_seqs(y1)])

                seq_len = min([len(seq1), len(seq2)])
                dtw_val, path = fastdtw.fastdtw(seq1, seq2, radius=round(seq_len*0.33))
                #cos_val = cosine(seq1, seq2)
                dtw_total += dtw_val/len(path)
                N += 1
            if N:
                if (dtw_total/N < thresh_dtw) and True:#(cos_val < thresh_cos):
                    update_dict(group_match_dict, i, j)
                    update_dict(group_match_dict, j, i)

    all_groups_ix = matches_dict_to_groups(group_match_dict)
    all_groups = [[x for i in group for x in all_groups[i]] for group in all_groups_ix]
    #all_groups = [remove_group_duplicates(g, 0.95) for g in all_groups]
    
    return all_groups
