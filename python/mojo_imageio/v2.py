from __future__ import annotations

from . import v3


def imread(uri, format=None, **kwargs):
    return v3.imread(uri, extension=format, **kwargs)


def imwrite(uri, im, format=None, **kwargs):
    return v3.imwrite(uri, im, extension=format, **kwargs)


imsave = imwrite


def mimread(uri, format=None, **kwargs):
    return [imread(uri, format=format, **kwargs)]


def mimwrite(uri, ims, format=None, **kwargs):
    images = list(ims)
    if len(images) != 1:
        raise ValueError("covered codecs accept exactly one image")
    return imwrite(uri, images[0], format=format, **kwargs)

