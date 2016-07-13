#!/usr/bin/env python
# -*- coding: utf8 -*-

from mido import MidiFile
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv

import numpy as np

#######
# Pianorolls dims are  :   TIME  *  PITCH


def get_total_num_tick(song_path):
    # Midi length should be written in a meta message at the beginning of the file,
    # but in many cases, lazy motherfuckers didn't write it...

    # Read a midi file and return a dictionnary {track_name : pianoroll}
    mid = MidiFile(song_path)

    # Parse track by track
    num_ticks = 0
    for i, track in enumerate(mid.tracks):
        tick_counter = 0
        for message in track:
            # Note on
            time = float(message.time)
            tick_counter += time
        num_ticks = max(num_ticks, tick_counter)
    return num_ticks


def get_time(song_path, quantization):
    mid = MidiFile(song_path)
    # Tick per beat
    ticks_per_beat = mid.ticks_per_beat
    # Total number of ticks
    total_num_tick = get_total_num_tick(song_path)

    # Dimensions of the pianoroll for each track
    T_pr = int((total_num_tick / ticks_per_beat) * quantization)

    return T_pr


def read_midi(song_path, quantization):
    # Read a midi file and return a dictionnary {track_name : pianoroll}
    mid = MidiFile(song_path)
    # Tick per beat
    ticks_per_beat = mid.ticks_per_beat

    T_pr = get_time(song_path, quantization)
    N_pr = 128
    pianoroll = {}

    def add_note_to_pr(note_off, notes_on, pr):
        pitch_off, _, time_off = note_off
        # Note off : search for the note in the list of note on,
        # get the start and end time
        # write it in th pr
        match_list = [(ind, item) for (ind, item) in enumerate(notes_on) if item[0] == pitch_off]

        if len(match_list) == 0:
            raise Exception("Try to note off a note that has never been turned on")

        # Add note to the pr
        pitch, velocity, time_on = match_list[0][1]
        pr[time_on:time_off, pitch] = velocity
        # Remove the note from notes_on
        ind_match = match_list[0][0]
        del notes_on[ind_match]
        return

    # Parse track by track
    counter_unnamed_track = 0
    for i, track in enumerate(mid.tracks):
        # Instanciate the pianoroll
        pr = np.zeros([T_pr, N_pr])
        time_counter = 0
        notes_on = []
        for message in track:
            print message
            # Time. Must be incremented, whether it is a note on/off or not
            time = float(message.time)
            time_counter += time / ticks_per_beat * quantization
            # Time in pr (mapping)
            time_pr = int(time_counter)
            # Note on
            if message.type == 'note_on':
                # Get pitch
                pitch = message.note
                # Get velocity
                velocity = message.velocity
                if velocity > 0:
                    notes_on.append((pitch, velocity, time_pr))
                elif velocity == 0:
                    add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)
            # Note off
            elif message.type == 'note_off':
                pitch = message.note
                velocity = message.velocity
                add_note_to_pr((pitch, velocity, time_pr), notes_on, pr)

        # We deal with discrete values ranged between 0 and 127
        #     -> convert to int
        pr = pr.astype(np.int16)
        if np.sum(np.sum(pr)) > 0:
            name = track.name
            if name == u'':
                name = 'unnamed' + str(counter_unnamed_track)
                counter_unnamed_track += 1
            if name in pianoroll.keys():
                # Take max of the to pianorolls (lame solution;...)
                pianoroll[name] = np.maximum(pr, pianoroll[name])
            else:
                pianoroll[name] = pr

    return pianoroll


if __name__ == '__main__':
    song_path = 'testt.mid'
    pianoroll = read_midi(song_path, 12)
    for name_instru in pianoroll.keys():
        np.savetxt(name_instru + '.csv', pianoroll[name_instru], delimiter=',')
        dump_to_csv(name_instru + '.csv', name_instru + '.csv')
