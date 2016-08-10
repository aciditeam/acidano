#!/usr/bin/env python
# -*- coding: utf8 -*-

import math
import numpy as np
from pianoroll_processing import pitch_class, sum_along_instru_dim, get_pianoroll_time
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from event_level import get_event_ind

import needleman_chord


def linear_warp_pr(pianoroll, T_target):
    # Ensure we actually read a pianoroll
    out = {}
    T_source = get_pianoroll_time(pianoroll)
    ratio = T_source / float(T_target)
    index_mask = [int(math.floor(x * ratio)) for x in range(0, T_target)]
    for k,v in pianoroll.iteritems():
        out[k] = pianoroll[k][index_mask, :]
    return out


# Must convert this to a list of integer
def conversion_to_integer_list(pr, lenght):
    nb_class = pr.shape[1]
    mask = np.zeros((nb_class), dtype=np.int)
    for x in range(nb_class):
        mask[x] = int(2**x)
    output = np.sum(pr * mask, axis=1)
    return output.tolist()


def needleman_chord_wrapper(pr1, pr2):
    pr1_pitch_class = pitch_class(pr1)
    pr2_pitch_class = pitch_class(pr2)
    len1 = pr1_pitch_class.shape[0]
    len2 = pr2_pitch_class.shape[0]

    # Compute the trace
    pr1_list = conversion_to_integer_list(pr1_pitch_class, len1)
    pr2_list = conversion_to_integer_list(pr2_pitch_class, len2)
    trace_1, trace_2 = needleman_chord.needleman_chord(pr1_list, pr2_list, 0, 0)

    return trace_1, trace_2


# Warp PR
def warp_pr_aux(pr, path):
    pr_warp = {}
    for k, v in pr.iteritems():
        pr_warp[k] = v[path]
    return pr_warp


def dtw_pr(pr0, pr1):
    # Flatten pr to compute the path
    pr0_flat = sum_along_instru_dim(pr0)
    pr1_flat = sum_along_instru_dim(pr1)

    def fun_thresh(y):
        return np.minimum(y,1).astype(int)

    distance, path = fastdtw(pr0_flat, pr1_flat, dist=lambda a,b: euclidean(fun_thresh(a), fun_thresh(b)))
    # Get paths
    path0 = [e[0] for e in path]
    path1 = [e[1] for e in path]

    pr0_warp = warp_pr_aux(pr0, path0)
    pr1_warp = warp_pr_aux(pr1, path1)

    return pr0_warp, pr1_warp


def needleman_event_chord_wrapper(pr0_dict, pr1_dict):
    # Event level
    # Pitch-class
    # Needleman-Wunsch

    pr0 = sum_along_instru_dim(pr0_dict)
    pr1 = sum_along_instru_dim(pr1_dict)

    # Get longest sequence
    if pr0.shape[0] < pr1.shape[0]:
        pr_long = pr1
        pr_short = pr0
        pr_short_dict = pr0_dict
        short_ind = 0
    else:
        pr_long = pr0
        pr_short = pr1
        pr_short_dict = pr1_dict
        short_ind = 1

    # Event level
    el_long = get_event_ind(pr_long)
    el_short = get_event_ind(pr_short)
    pr_long_el = pr_long[el_long,:]
    pr_short_el = pr_short[el_short,:]

    # Needleman-Wunsch wrapper (pitch-class before calling C function)
    trace_long, trace_short = needleman_chord_wrapper(pr_long_el, pr_short_el)

    # Write down the time marker
    time_marker = [(0,0)]
    counter_long = 0
    counter_short = 0
    for bool_long, bool_short in zip(trace_long, trace_short):
        if bool_long and bool_short:
            time_marker.append((el_short[counter_short], el_long[counter_long]))
            counter_long += 1
            counter_short += 1
        elif bool_long:
            counter_long += 1
        elif bool_short:
            counter_short += 1

    # Add last indices
    time_marker.append((len(pr_short), len(pr_long)))

    # From time marker, create a time path
    def build_indices(s0, t0, s1, t1):
        ratio = (s1-s0)/float(t1-t0)
        l = [int(round(ratio * e + s0)) for e in range(0, t1-t0)]
        return l

    path = []
    for (s0, t0), (s1, t1) in zip(time_marker[:-1], time_marker[1:]):
        path.extend(build_indices(s0, t0, s1, t1))
    # Shortest sequence is linearly wrapped to the longest sequence
    pr_short_warped = warp_pr_aux(pr_short_dict, path)

    return (pr_short_warped, pr1_dict) \
        if short_ind == 0 \
        else (pr0_dict, pr_short_warped)


if __name__ == '__main__':
    l1 = [1,2,1,2,3,4,3,4,5,6]
    l2 = [1,2,3,4,5,6]

    struct = needleman_chord.needleman_chord(l1, l2, 0, 0)

    # arr1 = np.asarray(l1)
    # arr2 = np.asarray(l2)

    ind1 = struct[0]
    ind2 = struct[1]

    arr1 = np.zeros((len(ind1)))
    arr2 = np.zeros((len(ind2)))
    # len(ind1) and len(ind2) should be the same
    counter = 0
    for i, elem1 in enumerate(ind1):
        try:
            if elem1:
                arr1[i] = l1[counter]
                counter += 1
            else:
                arr1[i] = -1
        except:
            import pdb; pdb.set_trace()
    counter = 0
    for i, elem2 in enumerate(ind2):
        try:
            if elem2:
                arr2[i] = l2[counter]
                counter += 1
            else:
                arr2[i] = -1
        except:
            import pdb; pdb.set_trace()
    print ("Original sequences :")
    print(l1)
    print(l2)

    print ("Aligned sequences :")
    # print(arr1[ind1])
    # print(arr2[ind2])
    print(arr1)
    print(arr2)
