import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.cm as cm


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

# ##### MSCOCO ##### #
# How to: choose a colormap, generate a linspace from 0 to 1
# with nclasses bins as follows
# ATTENTION: 1000 classes are too many for one single colormap..
# some colors are the same, because the transition is too smooth.
# we should join more colormaps!

nclasses = 1000
color_bins = np.linspace(0, 1, nclasses)
norm = mpl.colors.Normalize(vmin=0, vmax=1)
m = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap('Pastel2'))
color_labels_coco = m.to_rgba(color_bins)[:, :3]


# ##### DAIMLER ##### #
color_labels_daimler = OrderedDict([
    (0, np.array([128, 64, 128], dtype=np.uint8)),  # ground
    (1, np.array([64, 0, 128], dtype=np.uint8)),  # vehicle
    (2, np.array([64, 64, 0], dtype=np.uint8)),  # pedestrian
    (3, np.array([128, 128, 128], dtype=np.uint8)),  # sky
    (4, np.array([128, 0, 0], dtype=np.uint8)),  # building
    (5, np.array([0, 0, 0], dtype=np.uint8))  # Unlabeled
    ])

headers_daimler = ["Ground", "Vehicle", "Pedestrian", "Sky",
                   "Building", "Void"]

# ##### KITTI ROAD ##### #
color_labels_kitti_road = OrderedDict([
    (0, np.array([128, 64, 128], dtype=np.uint8)),  # non road
    (1, np.array([64, 0, 128], dtype=np.uint8)),  # road
    (2, np.array([0, 0, 0], dtype=np.uint8))  # Unlabeled
    ])

headers_kitti_road = ["Road", "Non-road", "Void"]

# ##### CAMVID ##### #
color_labels_camvid = OrderedDict([
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

# ##### HORSES ##### #
color_labels_horses = OrderedDict([
    (0, np.array([255, 255, 255], dtype=np.uint8)),  # Horse
    (1, np.array([0, 0, 0], dtype=np.uint8))  # Unlabeled
    ])

headers_horses = ["Horses", "Non-horses"]


# DATASET DICTIONARIES #
color_labels_datasets = dict()
color_labels_datasets["camvid"] = color_labels_camvid
color_labels_datasets["daimler"] = color_labels_daimler
color_labels_datasets["kitti_road"] = color_labels_kitti_road
color_labels_datasets["horses"] = color_labels_horses

color_list_datasets = dict()
for key, value in color_labels_datasets.iteritems():
    color_list_datasets[key] = np.asarray(
                    [z for z in zip(*value.items())[1]]) / 255.

headers_datasets = dict()
headers_datasets["camvid"] = headers_camvid
headers_datasets["daimler"] = headers_daimler
headers_datasets["kitti_road"] = headers_kitti_road
headers_datasets["horses"] = headers_horses
