#!/usr/bin/env python
# -*- coding: utf8 -*-

from collections import OrderedDict
import cPickle as pickle


# Dico with the regexp for intru_mapping
def buildDico():
    dico = {
        u"bassoons": [
            u"bassoon",
            u"bsn\."
        ],
        u"clarinets": [
            u"clarinet",
            u"clar\."
        ],
        u"double basses": [
            u"bass",
            u"db\."
        ],
        u"flutes": [
            u"flute"
        ],
        u"harps": [
            u"harp"
        ],
        u"horns": [
            u"horn",
            u"hn\.",
            u"cor"
        ],
        u"oboes": [
            u"oboe",
            u"ob\.",
            u"hautbois"
        ],
        u"others": [
            u"wind",
            u"brass",
            u"string"
        ],
        u"percussions": [
            u"percu",
            u"perc"
        ],
        u"piano": [
            u"piano",
            u"reduction",
            u"r\u00e9d"
        ],
        u"piccolos": [
            u"piccolo"
        ],
        u"timpani": [
            u"timpani",
            u"timbal"
        ],
        u"trombones": [
            u"trombone",
            u"tr\.",
            u"tbn\."
        ],
        u"trumpets": [
            u"tr[uo]mpet",
            u"tpt\."
        ],
        u"tubas": [
            u"tuba"
        ],
        u"violas": [
            u"violas?",
            u"vla\.",
            u"alti"
        ],
        u"violins i": [
            u"viol[io]ns?(\s)*i[^i]*$",
            u"viol[io]ns?(\s)*1",
            u"vl\.(\s)*i[^i]*$"
        ],
        u"violins ii": [
            u"viol[io]ns?(\s)*ii",
            u"viol[io]ns?(\s)*2",
            u"vl\.(\s)*ii"
        ],
        u"violoncellos": [
            u"cello",
            u"celli",
            u"vc\."
        ]
    }
    dico_ordered = OrderedDict()
    for key, value in sorted(dico.items()):
        dico_ordered[key] = value
    return dico_ordered

if __name__ == '__main__':
    dico = buildDico()
    pickle.dump(dico, open("instru_regex.p", "wb"))
