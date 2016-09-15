#!/usr/bin/env python
# -*- coding: utf8 -*-

from acidano.visualization.numpy_array.write_numpy_array_html import write_numpy_array_html
from mido import MidiFile
from write_midi import write_midi
from acidano.data_processing.utils.pianoroll_processing import sum_along_instru_dim
from acidano.visualization.numpy_array.dumped_numpy_to_csv import dump_to_csv
from acidano.data_processing.utils.pianoroll_processing import get_pianoroll_time, clip_pr
from acidano.data_processing.utils.time_warping import linear_warp_pr
from subprocess import call

import numpy as np

#######
# Pianorolls dims are  :   TIME  *  PITCH


class Read_midi(object):
    def __init__(self, song_path, quantization):
        ## Metadata
        self.__song_path = song_path
        self.__quantization = quantization

        ## Pianoroll
        self.pianoroll = None
        self.__T_pr = None

        ## Private misc
        self.__num_ticks = None
        self.__T_file = None

    @property
    def quantization(self):
        return self.__quantization

    @property
    def T_pr(self):
        return self.__T_pr

    @property
    def T_file(self):
        return self.__T_file

    @property
    def pianoroll(self):
        return self.__pianoroll

    @pianoroll.setter
    def pianoroll(self, pr):
        # Ensure that the dimensions are always correct
        if pr is None:
            self.__pianoroll = None
            return
        T_pr = get_pianoroll_time(pr)
        if T_pr:
            self.__T_pr = T_pr
            self.__pianoroll = pr
        else:
            self.__pianoroll = None

    def get_total_num_tick(self):
        # Midi length should be written in a meta message at the beginning of the file,
        # but in many cases, lazy motherfuckers didn't write it...

        # Read a midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)

        # Parse track by track
        num_ticks = 0
        for i, track in enumerate(mid.tracks):
            tick_counter = 0
            for message in track:
                # Note on
                time = float(message.time)
                tick_counter += time
            num_ticks = max(num_ticks, tick_counter)
        self.__num_ticks = num_ticks

    def get_pitch_range(self):
        mid = MidiFile(self.__song_path)
        min_pitch = 200
        max_pitch = 0
        for i, track in enumerate(mid.tracks):
            for message in track:
                if message.type in ['note_on', 'note_off']:
                    pitch = message.note
                    if pitch > max_pitch:
                        max_pitch = pitch
                    if pitch < min_pitch:
                        min_pitch = pitch
        return min_pitch, max_pitch

    def get_time_file(self):
        # Get the time dimension for a pianoroll given a certain quantization
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat
        # Total number of ticks
        self.get_total_num_tick()
        # Dimensions of the pianoroll for each track
        self.__T_file = int((self.__num_ticks / ticks_per_beat) * self.__quantization)
        return self.__T_file

    def read_file(self):
        # Read the midi file and return a dictionnary {track_name : pianoroll}
        mid = MidiFile(self.__song_path)
        # Tick per beat
        ticks_per_beat = mid.ticks_per_beat

        # Get total time
        self.get_time_file()
        T_pr = self.__T_file
        # Pitch dimension
        N_pr = 128
        pianoroll = {}

        def add_note_to_pr(note_off, notes_on, pr):
            pitch_off, _, time_off = note_off
            # Note off : search for the note in the list of note on,
            # get the start and end time
            # write it in th pr
            match_list = [(ind, item) for (ind, item) in enumerate(notes_on) if item[0] == pitch_off]
            if len(match_list) == 0:
                print("Try to note off a note that has never been turned on")
                # Do nothing
                return

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
                # print message
                # Time. Must be incremented, whether it is a note on/off or not
                time = float(message.time)
                time_counter += time / ticks_per_beat * self.__quantization
                # Time in pr (mapping)
                time_pr = int(round(time_counter))
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
                    # Take max of the to self.pianorolls (lame solution;...)
                    pianoroll[name] = np.maximum(pr, pianoroll[name])
                else:
                    pianoroll[name] = pr
        self.pianoroll = pianoroll
        return pianoroll


if __name__ == '__main__':
    song_path = '/Users/leo/Recherche/GitHub_Aciditeam/database/Solo_midi/Nottingham/train/ashover_simple_chords_1.mid'
    quantization = 2
    midifile = Read_midi(song_path, quantization)
    pr = midifile.read_file()

    AAA = sum_along_instru_dim(pr)
    # AAA = AAA[0:quantization*12, :]
    AAA = AAA[:100,21:109]
    np.savetxt('DEBUG/temp.csv', AAA, delimiter=',')
    dump_to_csv('DEBUG/temp.csv', 'DEBUG/temp.csv')
    write_numpy_array_html("DEBUG/pr_aligned.html", "temp")
    # call(["open", "DEBUG/numpy_vis.html"])

    # pr_clip = clip_pr(pr)
    # pr_warped = linear_warp_pr(pr_clip, int(midifile.T_pr * 0.6))

    # write_midi(pr_warped, midifile.quantization, 'DEBUG/out.mid')
    # write_midi(midifile.pianoroll, midifile.quantization, 'DEBUG/out2.mid')
    # for name_instru in midifile.pianoroll.keys():
    #     np.savetxt('DEBUG/' + name_instru + '.csv', midifile.pianoroll[name_instru], delimiter=',')
    #     dump_to_csv('DEBUG/' + name_instru + '.csv', 'DEBUG/' + name_instru + '.csv')
