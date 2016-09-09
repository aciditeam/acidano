#!/bin/zsh

# gcc needleman_chord.c
# ./a.out
# atom file2.txt
rm -r build
rm ../needleman_chord.so
python setup.py build
cp build/lib.macosx-10.11-intel-2.7/needleman_chord.so ..
# cd ..
# python time_warping.py
