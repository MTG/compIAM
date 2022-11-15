from collections import Counter

import textgrid
import numpy as np
import pandas as pd

import math


def load_annotations_new(annotations_path, min_m=None, max_m=None):

    annotations_orig = pd.read_csv(annotations_path, sep='\t')
    annotations_orig.columns = ['tier', 'not_used', 's1', 's2', 'duration', 'text']
    
    annotations_orig['s1'] = pd.to_datetime(annotations_orig['s1']).apply(lambda y: y.time())
    annotations_orig['s2'] = pd.to_datetime(annotations_orig['s2']).apply(lambda y: y.time())
    annotations_orig['duration'] = pd.to_datetime(annotations_orig['duration']).apply(lambda y: y.time())

    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['duration'] = annotations_orig['duration'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    
    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: round(y,1))
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: round(y,1))
    

    if min_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)>=min_m]
    if max_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)<=max_m]


    annotations_underlying = annotations_orig[annotations_orig['tier'].isin(['underlying_full_phrase','underlying_sancara', 'root_full_phrase','root_sancara'])]
    annotations_true = annotations_orig[annotations_orig['tier'].isin(['full_phrase','sancara'])]

    annotations_merge = annotations_underlying.merge(annotations_true, on=['s1','s2'], suffixes=('_u','_t'), how='left')

    annotations_merge.columns = ['tier', 'not_used_u', 's1', 's2', 'duration', 'text', 'tier_t', 'not_used_t', 'duration_t', 'text_full']
    annotations_merge = annotations_merge[['tier', 's1', 's2', 'duration', 'text', 'text_full']]

    good_text = [k for k,v in Counter(annotations_merge['text']).items() if v>1]
    annotations_good = annotations_merge[annotations_merge['text'].isin(good_text)]

    #annotations_good = annotations_good[annotations_good['s2']- annotations_good['s1']>=1]
    annotations_good['tier'] = annotations_good['tier'].apply(lambda y: y.replace('root','underlying'))

    annotations_good = annotations_good.groupby(['s1','s2']).first().reset_index()

    return annotations_good[['tier', 's1', 's2', 'text', 'text_full']]


def load_annotations_brindha(annotations_path, min_m=None, max_m=None):

    annotations_orig = pd.read_csv(annotations_path, sep='\t')
    annotations_orig.columns = ['tier', 'not_used', 's1', 's2', 'duration', 'text']
    
    annotations_orig['s1'] = pd.to_datetime(annotations_orig['s1']).apply(lambda y: y.time())
    annotations_orig['s2'] = pd.to_datetime(annotations_orig['s2']).apply(lambda y: y.time())
    annotations_orig['duration'] = pd.to_datetime(annotations_orig['duration']).apply(lambda y: y.time())

    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)
    annotations_orig['duration'] = annotations_orig['duration'].apply(lambda y: y.hour*120 + y.minute*60 + y.second + y.microsecond*10e-7)

    if min_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)>=min_m]
    if max_m:
        annotations_orig = annotations_orig[annotations_orig['duration'].astype(float)<=max_m]

    annotations_orig = annotations_orig[annotations_orig['tier'].isin(['underlying_full_phrase','underlying_sancara', 'root_full_phrase','root_sancara'])]
    good_text = [k for k,v in Counter(annotations_orig['text']).items() if v>1]
    annotations_orig = annotations_orig[annotations_orig['text'].isin(good_text)]

    #annotations_orig = annotations_orig[annotations_orig['s2']- annotations_orig['s1']>=1]
    annotations_orig['tier'] = annotations_orig['tier'].apply(lambda y: y.replace('root','underlying'))
    
    # remove duplicates
    annotations_orig['s1'] = annotations_orig['s1'].apply(lambda y: round(y,1))
    annotations_orig['s2'] = annotations_orig['s2'].apply(lambda y: round(y,1))
    annotations_orig = annotations_orig.groupby(['s1','s2']).first().reset_index()

    # if s1 and s2
    return annotations_orig[['tier', 's1', 's2', 'text']]



def get_coverage(pitch, starts_seq_exc, lengths_seq_exc):
    pitch_coverage = pitch.copy()
    pitch_coverage[:] = 0

    for i, group in enumerate(starts_seq_exc):
        for j, s in enumerate(group):
            l = lengths_seq_exc[i][j]
            pitch_coverage[s:s+l] = 1

    return np.sum(pitch_coverage)/len(pitch_coverage)

def is_match_v2(sp, lp, sa, ea, partial_perc=0.3):

    ep = sp + lp
    
    # partial if identified pattern captures a
    # least <partial_perc> of annotation
    la = (ea-sa) # length of annotation

    overlap = 0

    # pattern starts in annotation
    if (sa <= sp <= ea):
        # and ends in annotation
        if ep < ea:
            overlap = (ep-sp)
        # and ends after annotation
        else:
            overlap = (ea-sp)

    # pattern ends in annotation
    if (sa <= ep <= ea):
        # and starts before annotation
        if sa < sp:
            overlap = (ep-sp)
        # and starts in annotation
        else:
            overlap = (ep-sa)

    # pattern contains annotation entirely
    if (sp <= sa) and (ea <= ep):
        overlap = la

    # if intersection between annotation and returned pattern is 
    # >= <partial_perc> of each its a match!
    if overlap/la >= partial_perc and overlap/lp >= partial_perc:
        return 'match'
    else:
        return None


def evaluate_annotations(annotations_raw, starts, lengths, partial_perc):
    annotations = annotations_raw.copy()
    results_dict = {}
    group_num_dict = {}
    occ_num_dict = {}
    is_matched_arr = []
    for i, seq_group in enumerate(starts):
        ima = []
        for j, seq in enumerate(seq_group):
            im = 0
            length = lengths[i][j]
            for ai, (tier, s1, s2, text, text_true) in zip(annotations.index, annotations.values):
                matched = is_match_v2(seq, length, s1, s2, partial_perc=partial_perc)
                if matched:
                    im = 1
                    if ai not in results_dict:
                        results_dict[ai] = matched
                        group_num_dict[ai] = i
                        occ_num_dict[ai] = j
            ima = ima + [im]
        is_matched_arr.append(ima)

    annotations['match']     = [results_dict[i] if i in results_dict else 'no match' for i in annotations.index]
    annotations['group_num'] = [group_num_dict[i] if i in group_num_dict else None for i in annotations.index]
    annotations['occ_num']   = [occ_num_dict[i] if i in occ_num_dict else None for i in annotations.index]

    return annotations, is_matched_arr


def evaluate(annotations_raw, starts, lengths, partial_perc):
    annotations, is_matched = evaluate_annotations(annotations_raw, starts, lengths, partial_perc)
    ime = [x for y in is_matched for x in y]
    precision = sum(ime)/len(ime) if ime else 1
    if len(annotations)>0:
        recall = sum(annotations['match']!='no match')/len(annotations)
    else:
        recall = np.nan
    f1 = f1_score(precision, recall)
    return recall, precision, f1, annotations


def f1_score(p,r):
    return 2*p*r/(p+r) if (p+r != 0) else 0


def get_grouping_accuracy(annotations_tagged):
    """
    Compute grouping accuracy - how often is each matched pattern grouped to
    the "correct" group?

    and group distribution - # assigned groups/ # unique matched patterns

    Returns: grouping_accuracy, group distribution
    """
    matched = annotations_tagged[annotations_tagged['match']=='match']

    # group to get pattern: list of groups it is in
    grouped = matched.groupby('text')['group_num'].apply(list).reset_index()
    grouped.columns = ['text', 'all_groups']

    # assign group number to pattern corresponding to most populated group
    get_nom = lambda g: sorted(Counter(g).items(), key=lambda y: -y[1])[0][0]
    grouped['nominated_group'] = grouped['all_groups'].apply(get_nom)

    # calculate grouping accuracy
    # how often is this pattern in the correct group?
    numerator = 0   
    denominator = 0
    for i,pattern in grouped.iterrows():
        nom = pattern['nominated_group']
        all_groups = pattern['all_groups']
        pos = len([x for x in all_groups if x==nom])
        neg = len(all_groups)-pos
        numerator += pos
        denominator += pos+neg

    if denominator:
        grouping_accuracy = numerator/denominator
    else:
        grouping_accuracy = 0

    if len(grouped):
        distribution = len(set(grouped['nominated_group']))/len(grouped)
    else:
        distribution = 0

    return grouping_accuracy, distribution
    