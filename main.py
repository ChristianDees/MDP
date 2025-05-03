#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: Christian Dees & Aitiana Mondragon
"""


# (currState, action): [probability, reward, nextState]
transitions = {
    ('RU8p', 'P'): [(1.0, 2, 'TU10p')],
    ('RU8p', 'R'): [(1.0, 0, 'RU10p')],
    ('RU8p', 'S'): [(1.0, -1, 'RD10p')],
    ('TU10p', 'P'): [(1.0, 2, 'TU10a')],
    ('TU10p', 'R'): [(1.0, 0, 'RU8a')],
    ('RU10p', 'P'): [(0.5, 2, 'RU8a'), (0.5, 2, 'TU10a')],
    ('RU10p', 'R'): [(1.0, 0, 'RU8a')],
    ('RU10p', 'S'): [(1.0, -1, 'RD8a')],
    ('RD10p', 'P'): [(0.5, 2, 'RD8a'), (0.5, 2, 'TD10a')],
    ('RD10p', 'R'): [(1.0, 0, 'RD8a')],
    ('RU8a', 'P'): [(1.0, 2, 'RU10a')],
    ('RU8a', 'R'): [(1.0, 0, 'RU10a')],
    ('RU8a', 'S'): [(1.0, -1, 'RD10a')],
    ('RD8a', 'P'): [(1.0, 2, 'RD10a')],
    ('RD8a', 'R'): [(1.0, 0, 'RD10a')],
    ('TU10a', 'any'): [(1.0, -1, 'terminal')],
    ('RU10a', 'any'): [(1.0, 0, 'terminal')],
    ('RD10a', 'any'): [(1.0, 4, 'terminal')],
    ('TD10a', 'any'): [(1.0, 3, 'terminal')]
}