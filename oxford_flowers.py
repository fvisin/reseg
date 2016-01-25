import os

from scipy.io import loadmat
import numpy
from PIL import Image

from zca import FWhitening


def properties():
    return {}
    # 'reshape': [212, 264, 3],
    # 'reorder': [0, 1, 2],
    # 'rereorder': [0, 1, 2]


def load_data(
        # path=os.path.expanduser('~/arctic_data/oxford_flowers'),
        path=os.path.expanduser('/Tmp/visin/oxford_flowers'),
        randomize=False,
        resize_images=True,
        resize_size=-1,
        zero_pad=True,
        crop=False,
        color=True,
        whiten=False,
        split=[.44, .22],
        which_split=0,
        with_filenames=False,
        with_fullmasks=False,
        rng=None,
        **kwargs):
    """Dataset loader

    Parameter
    ---------
    path : string
        the path to the dataset images.
            splits = loadmat(
    randomize
    resize
    zero_pad
    crop
    out_name
    chunksize

    """
    #############
    # LOAD DATA #
    #############
    im_list_path = os.path.join(path, "trimaps",
                                "imlist.mat")
    im_ids = loadmat(im_list_path)["imlist"][0]
    # These two are *specifically* excluded due to invalid masks
    # image_1269.png and image_1270.png
    im_ids = [im for im in im_ids if im not in [1269, 1270]]
    im_path = os.path.join(path, 'jpg')

    def get_im_id(im):
        return int(im.split(".")[0].split("_")[-1])

    # list every files in dir and subdirs
    print "Processing the data ..."
    filenames = []
    for directory, _, images in os.walk(im_path):
        jpgs = [im for im in images if ".jpg" in im]
        matching_jpgs = [j for j in jpgs if get_im_id(j) in im_ids]
        filenames.extend(matching_jpgs)
    filenames = sorted(filenames)
    # v : k are swapped to look up array index for id
    matching_ids = [get_im_id(im) for im in filenames]
    im_ids_lookup = {v: k for k, v in enumerate(matching_ids)}

    if randomize:
        raise ValueError("DO NOT RANDOMIZE - USE PREDEFINED SPLIT")
        if rng is None:
            print('No rng was provided. Instantiating a new one...')
            numpy.random.RandomState(0xbeef)
        rng.shuffle(filenames)

    if resize_images and resize_size == -1:
        # Compute the mean height and mean width
        from operator import add
        resize_size = [0, 0]
        for f in filenames:
            f_sub = f.split(".")[0]
            mask = Image.open(os.path.join(path, 'trimaps',
                                           f_sub + ".png")).copy()
            resize_size = map(add, resize_size, mask.size)
        resize_size = [resize_size[0]/len(filenames),
                       resize_size[1]/len(filenames)]
        print('Image properties: w={}, h={}, reshape={}'.format(
            resize_size[0], resize_size[1], [
                resize_size[1], resize_size[0], (3 if color else 1)]))

    images = []
    masks = []
    fullmasks = []
    cropped_px = 0
    for f in filenames:
        im = Image.open(os.path.join(im_path, f)).copy()
        f_sub = f.split(".")[0]
        mask = Image.open(os.path.join(path, 'trimaps',
                                       f_sub + ".png")).copy()
        size = mask.size
        if with_fullmasks:
            fullmask = mask.convert('L')
            fullmask = numpy.array(mask.copy()).astype(numpy.uint8)
            max_class = 38
            fullmask[fullmask != max_class] = 0
            fullmask[fullmask == max_class] = 1
            fullmasks.append(fullmask)

        # RGB have different size than the mask..who knows why
        im = im.resize(size, Image.ANTIALIAS)

        if resize_images:
            rx, ry = resize_size
            if crop:
                # resize (keeping proportions)
                [x, y] = im.size
                dx = float(rx)/x
                dy = float(ry)/y
                ratio = max(dx, dy)
                x = int(x * ratio)
                y = int(y * ratio)

                im = im.resize((x, y), Image.ANTIALIAS)
                mask = mask.resize((x, y), Image.NEAREST)

                # crop
                if x != rx:
                    excess = (x - rx) / 2
                    im = im.crop((excess, 0, rx+excess, ry))
                    mask = mask.crop((excess, 0, rx+excess, ry))
                elif y != ry:
                    excess = (y - ry) / 2
                    im = im.crop((0, excess, rx, ry+excess))
                    mask = mask.crop((0, excess, rx, ry+excess))
                cropped_px += excess*2
            elif zero_pad:
                # resize (keeping proportions)
                [x, y] = im.size
                dx = float(rx)/x
                dy = float(ry)/y
                ratio = min(dy, dx)
                x = int(x * ratio)
                y = int(y * ratio)

                im = im.resize((x, y), Image.ANTIALIAS)
                mask = mask.resize((x, y), Image.NEAREST)

                tmp = im
                im = Image.new("RGB", (rx, ry))
                im.paste(tmp, ((rx-x)/2, (ry-y)/2))
                tmp = mask
                mask = Image.new("L", (rx, ry))
                mask.paste(tmp, ((rx-x)/2, (ry-y)/2))
            else:
                # resize (not keeping proportions)
                im = im.resize((rx, ry), Image.ANTIALIAS)
                mask = mask.resize((rx, ry), Image.NEAREST)

            assert tuple(im.size) == tuple(resize_size)

        im = numpy.array(im).astype(numpy.uint8)
        mask = numpy.array(mask.copy()).astype(numpy.uint8)
        max_class = 38
        mask[mask != max_class] = 0
        mask[mask == max_class] = 1

        assert 0 <= numpy.min(im) < 255
        assert 0 < numpy.max(im) <= 255
        assert numpy.min(im) != numpy.max(im)
        # images 1269 and 1270 violate this
        assert numpy.min(mask) != numpy.max(mask)

        images.append(im)
        masks.append(mask)

    images = numpy.asarray(images)
    masks = numpy.asarray(masks)
    ntot = len(images)

    split_path = os.path.join(path, "datasplits.mat")
    split_mat = loadmat(split_path)
    if which_split == 0:
        train_id = split_mat["trn1"]
        val_id = split_mat["val1"]
        test_id = split_mat["tst1"]
    elif which_split == 1:
        train_id = split_mat["trn2"]
        val_id = split_mat["val2"]
        test_id = split_mat["tst2"]
    elif which_split == 2:
        train_id = split_mat["trn3"]
        val_id = split_mat["val3"]
        test_id = split_mat["tst3"]
    else:
        raise ValueError("Unsupported value for"
                         "which_split: got %s" % str(which_split))
    # Look up the correct index - but it seems their splits have invalid
    # files in them...
    train_idx = numpy.array([im_ids_lookup[im_id] for im_id in train_id.ravel()
                             if im_id in im_ids])
    val_idx = numpy.array([im_ids_lookup[im_id] for im_id in val_id.ravel()
                           if im_id in im_ids])
    test_idx = numpy.array([im_ids_lookup[im_id] for im_id in test_id.ravel()
                            if im_id in im_ids])

    def mkname(im_id):
        return "image_%4d.jpg" % im_id

    train_fnames = [mkname(im_id) for im_id in train_id.ravel()
                    if im_id in im_ids]
    val_fnames = [mkname(im_id) for im_id in val_id.ravel()
                  if im_id in im_ids]
    test_fnames = [mkname(im_id) for im_id in test_id.ravel()
                   if im_id in im_ids]

    print "Computing dataset statistics ..."
    if resize_images:
        # compute per pixel statistics
        mean = images.mean(axis=0)[numpy.newaxis, ...]
        std = numpy.maximum(images.std(axis=0), 1e-8)[numpy.newaxis, ...]
    else:
        max_size = [0, 0]
        for m in masks[val_idx]:
            max_size = map(max, max_size, m.shape)

        arr = numpy.ma.empty(max_size + [3, len(val_idx)])
        arr.mask = True
        for n, (m, i) in enumerate(zip(masks[val_idx], val_idx)):
            arr[:images[i].shape[0], :images[i].shape[1], :, n] = images[i]
        mean = arr.mean()
        std = arr.std()
    mean = mean.astype('float32')
    std = std.astype('float32')

    if whiten:
        im = FWhitening(mean=mean).transform(im)

    # split datasets
    train_set_x = numpy.array(images[train_idx])
    train_set_y = numpy.array(masks[train_idx])
    valid_set_x = numpy.array(images[val_idx])
    valid_set_y = numpy.array(masks[val_idx])
    test_set_x = numpy.array(images[test_idx])
    test_set_y = numpy.array(masks[test_idx])
    if with_fullmasks:
        fullmasks = numpy.asarray((numpy.array(fullmasks)[train_idx],
                                   numpy.array(fullmasks)[val_idx],
                                   numpy.array(fullmasks)[test_idx]))

    train = (train_set_x, train_set_y)
    valid = (valid_set_x, valid_set_y)
    test = (test_set_x, test_set_y)
    print "load_data Done!"
    print('Tot images:{} Train:{} Valid:{} Test:{}').format(
        ntot, len(train_set_x) - 1, len(valid_set_x) - 1,
        len(test_set_x) - 1)

    rval = [train, valid, test, mean, std]
    if with_filenames:
        rval.append([train_fnames, val_fnames, test_fnames])
    if with_fullmasks:
        rval.append(fullmasks)
    return rval

if __name__ == "__main__":
    load_data(resize_images=False, with_fullmasks=True)
