from collections import OrderedDict

import numpy as np


# COLORMAPS
cmaps = [('Perceptually Uniform Sequential',
         ['viridis', 'inferno', 'plasma', 'magma']),
         ('Sequential',     ['Blues', 'BuGn', 'BuPu',
                             'GnBu', 'Greens', 'Greys', 'Oranges', 'OrRd',
                             'PuBu', 'PuBuGn', 'PuRd', 'Purples', 'RdPu',
                             'Reds', 'YlGn', 'YlGnBu', 'YlOrBr', 'YlOrRd']),
         ('Sequential (2)', ['afmhot', 'autumn', 'bone', 'cool',
                             'copper', 'gist_heat', 'gray', 'hot',
                             'pink', 'spring', 'summer', 'winter']),
         ('Diverging',      ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr',
                             'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'Spectral',
                             'seismic']),
         ('Qualitative',    ['Accent', 'Dark2', 'Paired', 'Pastel1',
                             'Pastel2', 'Set1', 'Set2', 'Set3']),
         ('Miscellaneous',  ['gist_earth', 'terrain', 'ocean', 'gist_stern',
                             'brg', 'CMRmap', 'cubehelix',
                             'gnuplot', 'gnuplot2', 'gist_ncar',
                             'nipy_spectral', 'jet', 'rainbow',
                             'gist_rainbow', 'hsv', 'flag', 'prism'])]


# ##### CAMVID ##### #
colormap_camvid = OrderedDict([
    (0, np.array([128, 128, 128], dtype=np.uint8)),  # sky
    (1, np.array([128, 0, 0], dtype=np.uint8)),  # Building
    (2, np.array([192, 192, 128], dtype=np.uint8)),  # Pole
    (3, np.array([128, 64, 128], dtype=np.uint8)),  # Road
    (4, np.array([0, 0, 192], dtype=np.uint8)),  # Sidewalk
    (5, np.array([128, 128, 0], dtype=np.uint8)),  # Tree
    (6, np.array([192, 128, 128], dtype=np.uint8)),  # SignSymbol
    (7, np.array([64, 64, 128], dtype=np.uint8)),  # Fence
    (8, np.array([64, 0, 128], dtype=np.uint8)),  # Car
    (9, np.array([64, 64, 0], dtype=np.uint8)),  # Pedestrian
    (10, np.array([0, 128, 192], dtype=np.uint8)),  # Bicyclist
    (11, np.array([0, 0, 0], dtype=np.uint8))  # Unlabeled
    ])

headers_camvid = ["Sky", "Building", "Column_Pole", "Road", "Sidewalk",
                  "Tree", "SignSymbol", "Fence", "Car", "Pedestrian",
                  "Bicyclist", "Void"]

# DATASET DICTIONARIES #
colormap_datasets = dict()
colormap_datasets["camvid"] = colormap_camvid

for key, value in colormap_datasets.iteritems():
    colormap_datasets[key] = np.asarray(
                    [z for z in zip(*value.items())[1]]) / 255.

headers_datasets = dict()
headers_datasets["camvid"] = headers_camvid
