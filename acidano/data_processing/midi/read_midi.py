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
    num_ticks = []
    for i, track in enumerate(mid.tracks):
        tick_counter = {}
        for message in track:
            # Note on
            if message.type == 'note_on' or message.type =='note_off':
                channel = message.channel
                time = float(message.time)
                if channel not in tick_counter.keys():
                    tick_counter[channel] = 0
                tick_counter[channel] += time
        if len(tick_counter) != 0:  # Avoid metadata tracks
            num_ticks.append(max(tick_counter.values()))
    num_ticks = set(num_ticks)
    return max(num_ticks)


def read_midi(song_path, quantization):
    # Read a midi file and return a dictionnary {track_name : pianoroll}
    mid = MidiFile(song_path)

    # Tick per beat
    ticks_per_beat = mid.ticks_per_beat
    # Total number of ticks
    total_num_tick = get_total_num_tick(song_path)

    # Dimensions of the pianoroll for each track
    T_pr = int((total_num_tick / ticks_per_beat) * quantization)
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
    for i, track in enumerate(mid.tracks):
        # Instanciate the pianoroll
        pr = np.zeros([T_pr, N_pr])
        time_counter = {}
        notes_on = {}
        for message in track:
            # Note on
            def init_channel_time(channel):
                if channel not in time_counter.keys():
                    # The counter of a new channel has to be initialized at the value of the other counters
                    if len(time_counter) == 0:
                        time_counter[channel] = 0
                    else:
                        rand_key = time_counter.keys()[0]
                        time_counter[channel] = time_counter[rand_key]
                    return
            if message.type == 'note_on':
                # Get channel
                channel = message.channel
                # Get pitch
                pitch = message.note
                # Get velocity
                velocity = message.velocity
                # Time
                time = float(message.time)
                # Time counter for each channel
                init_channel_time(channel)
                time_counter[channel] += time / ticks_per_beat * quantization
                # Time in pr (mapping)
                time_pr = int(time_counter[channel])
                if velocity > 0:
                    if channel not in notes_on.keys():
                        notes_on[channel] = []
                    notes_on[channel].append((pitch, velocity, time_pr))
                elif velocity == 0:
                    if channel not in notes_on.keys():
                        raise Exception("First note of channel " + str(channel) + ' is a note off')
                    add_note_to_pr((pitch, velocity, time_pr), notes_on[channel], pr)
            # Note off
            elif message.type == 'note_off':
                channel = message.channel
                pitch = message.note
                velocity = message.velocity
                time = float(message.time)
                init_channel_time(channel)
                time_counter[channel] += (time / ticks_per_beat) * quantization
                time_pr = int(time_counter[channel])
                add_note_to_pr((pitch, velocity, time_pr), notes_on[channel], pr)

        if np.sum(np.sum(pr)) > 0:
            pianoroll[track.name] = pr

    return pianoroll


if __name__ == '__main__':
    song_path = 'test.mid'
    pianoroll = read_midi(song_path, 12)
    import pdb; pdb.set_trace()
    for name_instru in pianoroll.keys():
        np.savetxt(name_instru + '.csv', pianoroll[name_instru], delimiter=',')
        dump_to_csv(name_instru + '.csv', name_instru + '.csv')
