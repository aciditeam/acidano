#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

def write_waveform_html(filename, dataname, d3js_source_path=None):
    if d3js_source_path is None:
        user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
        for user_path in user_paths:
            if re.search('acidano', user_path):
                d3js_source_path = user_path + 'acidano/visualization/d3.v3.min.js'
                break
    return
