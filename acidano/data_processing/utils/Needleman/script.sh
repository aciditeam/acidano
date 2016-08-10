#!/bin/zsh

# gcc needleman_chord.c
# ./a.out
# atom file2.txt
python setup.py build
cp build/lib.macosx-10.6-x86_64-2.7/needleman_chord.so ..
cd ..
python time_warping.py
