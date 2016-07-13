#!/usr/bin/env python
# -*- coding: utf8 -*-

# Midi lib
import mido
from mido import MidiFile

import numpy as np

from read_midi import read_midi


def write_midi(pr, quantization, write_path, tempo=80):
    # Tempo
    microseconds_per_beat = mido.bpm2tempo(tempo)
    # Write a pianoroll in a midi file
    mid = MidiFile()
    # ticks_per_beat can be the quantization, this simplify the writing process
    mid.ticks_per_beat = quantization

    # Each instrument is a track
    for instrument_name, matrix in pr.iteritems():
        # A bit shity : if the pr is a binary pr, multiply by 127
        if np.max(matrix) == 1:
            matrix = matrix * 127
        # Add a new track with the instrument name to the midi file
        track = mid.add_track(instrument_name)
        # transform the matrix in a list of (pitch, velocity, time)
        events = pr_to_list(matrix)
        # Tempo
        track.append(mido.MetaMessage('set_tempo', tempo=microseconds_per_beat))
        # Write events in the midi file
        for event in events:
            pitch, velocity, time = event
            if velocity == 0:
                track.append(mido.Message('note_off', note=pitch, time=time))
            else:
                track.append(mido.Message('note_on', note=pitch, velocity=velocity, time=time))
    mid.save(write_path)
    return


def pr_to_list(pr):
    # List event = (pitch, velocity, time)
    T, N = pr.shape
    t_last = 0
    pr_tm1 = np.zeros(N)
    list_event = []
    for t in range(T):
        pr_t = pr[t]
        if (pr_t != pr_tm1).any():
            # mask all values in pr_t except the new ones
            mask = np.ma.getmask(np.ma.masked_where(pr_t != pr_tm1, pr_t))
            # print mask
            # import pdb; pdb.set_trace()
            for n in range(N):
                if mask[n]:
                    pitch = n
                    velocity = int(pr_t[n])
                    # Time is increment since last event
                    t_event = t - t_last
                    t_last = t
                    list_event.append((pitch, velocity, t_event))
        pr_tm1 = pr_t
    return list_event


if __name__ == '__main__':
    fpath = 'test.mid'
    quantization = 24
    pr = read_midi(fpath, quantization)
    write_midi(pr, quantization, 'out.mid', tempo=40)
