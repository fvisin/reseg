import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from collections import OrderedDict
import matplotlib.cm as cm


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
colormap_coco = m.to_rgba(color_bins)[:, :3]


# ##### DAIMLER ##### #
colormap_daimler = OrderedDict([
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
colormap_kitti_road = OrderedDict([
    (0, np.array([128, 64, 128], dtype=np.uint8)),  # non road
    (1, np.array([64, 0, 128], dtype=np.uint8)),  # road
    (2, np.array([0, 0, 0], dtype=np.uint8))  # Unlabeled
    ])

headers_kitti_road = ["Road", "Non-road", "Void"]

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

# ##### FASHIONISTA ##### #
colormap_fashionista = OrderedDict([
    (0, np.array([128, 128, 128], dtype=np.uint8)),
    (1, np.array([128, 0, 0], dtype=np.uint8)),
    (2, np.array([192, 192, 128], dtype=np.uint8)),
    (3, np.array([128, 64, 128], dtype=np.uint8)),
    (4, np.array([0, 0, 192], dtype=np.uint8)),
    (5, np.array([128, 128, 0], dtype=np.uint8)),
    (6, np.array([192, 128, 128], dtype=np.uint8)),
    (7, np.array([64, 64, 128], dtype=np.uint8)),
    (8, np.array([64, 0, 128], dtype=np.uint8)),
    (9, np.array([64, 64, 0], dtype=np.uint8)),
    (10, np.array([0, 128, 192], dtype=np.uint8)),
    (11, np.array([0, 0, 0], dtype=np.uint8)),
    (12, np.array([128, 128, 128], dtype=np.uint8)),
    (13, np.array([128, 0, 0], dtype=np.uint8)),
    (14, np.array([192, 192, 128], dtype=np.uint8)),
    (15, np.array([128, 64, 128], dtype=np.uint8)),
    (16, np.array([0, 0, 192], dtype=np.uint8)),
    (17, np.array([128, 128, 0], dtype=np.uint8)),
    (18, np.array([192, 128, 128], dtype=np.uint8)),
    (19, np.array([64, 64, 128], dtype=np.uint8)),
    (20, np.array([64, 0, 128], dtype=np.uint8)),
    (21, np.array([64, 64, 0], dtype=np.uint8)),
    (22, np.array([0, 128, 192], dtype=np.uint8)),
    (23, np.array([0, 0, 0], dtype=np.uint8)),
    (24, np.array([128, 128, 128], dtype=np.uint8)),
    (25, np.array([128, 0, 0], dtype=np.uint8)),
    (26, np.array([192, 192, 128], dtype=np.uint8)),
    (27, np.array([128, 64, 128], dtype=np.uint8)),
    (28, np.array([0, 0, 192], dtype=np.uint8)),
    (29, np.array([128, 128, 0], dtype=np.uint8)),
    (30, np.array([192, 128, 128], dtype=np.uint8)),
    (31, np.array([64, 64, 128], dtype=np.uint8)),
    (32, np.array([64, 0, 128], dtype=np.uint8)),
    (33, np.array([64, 64, 0], dtype=np.uint8)),
    (34, np.array([0, 128, 192], dtype=np.uint8)),
    (35, np.array([0, 0, 0], dtype=np.uint8)),
    (36, np.array([128, 128, 128], dtype=np.uint8)),
    (37, np.array([128, 0, 0], dtype=np.uint8)),
    (38, np.array([192, 192, 128], dtype=np.uint8)),
    (39, np.array([128, 64, 128], dtype=np.uint8)),
    (40, np.array([0, 0, 192], dtype=np.uint8)),
    (41, np.array([128, 128, 0], dtype=np.uint8)),
    (42, np.array([192, 128, 128], dtype=np.uint8)),
    (43, np.array([64, 64, 128], dtype=np.uint8)),
    (44, np.array([64, 0, 128], dtype=np.uint8)),
    (45, np.array([64, 64, 0], dtype=np.uint8)),
    (46, np.array([0, 128, 192], dtype=np.uint8)),
    (47, np.array([0, 0, 0], dtype=np.uint8)),
    (48, np.array([128, 128, 128], dtype=np.uint8)),
    (49, np.array([128, 0, 0], dtype=np.uint8)),
    (50, np.array([192, 192, 128], dtype=np.uint8)),
    (51, np.array([128, 64, 128], dtype=np.uint8)),
    (52, np.array([0, 0, 192], dtype=np.uint8)),
    (53, np.array([128, 128, 0], dtype=np.uint8)),
    (54, np.array([192, 128, 128], dtype=np.uint8)),
    (55, np.array([64, 64, 128], dtype=np.uint8)),
    (56, np.array([64, 0, 128], dtype=np.uint8)),
    ])


# ##### HORSES ##### #
colormap_flowers = OrderedDict([
    (0, np.array([128, 64, 128], dtype=np.uint8)),  # non road
    (1, np.array([64, 0, 128], dtype=np.uint8)),  # road
    ])


# ##### HORSES ##### #
colormap_horses = OrderedDict([
    (0, np.array([255, 255, 255], dtype=np.uint8)),  # Horse
    (1, np.array([0, 0, 0], dtype=np.uint8))  # Unlabeled
    ])

headers_horses = ["Horses", "Non-horses"]


# DATASET DICTIONARIES #
colormap_datasets = dict()
colormap_datasets["camvid"] = colormap_camvid
colormap_datasets["daimler"] = colormap_daimler
colormap_datasets["fashionista"] = colormap_fashionista
colormap_datasets["flowers"] = colormap_flowers
colormap_datasets["kitti_road"] = colormap_kitti_road
colormap_datasets["horses"] = colormap_horses

for key, value in colormap_datasets.iteritems():
    colormap_datasets[key] = np.asarray(
                    [z for z in zip(*value.items())[1]]) / 255.

headers_datasets = dict()
headers_datasets["camvid"] = headers_camvid
headers_datasets["daimler"] = headers_daimler
headers_datasets["kitti_road"] = headers_kitti_road
headers_datasets["horses"] = headers_horses
