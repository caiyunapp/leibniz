# -*- coding: utf-8 -*-

import numpy as np

import leibniz.core3d.basis.theta_phi_r as delegate


_name_ = 'lng,lat,alt'
_params_ = ('lng', 'lat', 'alt')


def d2r(d):
    return d * np.pi / 180


def r2d(r):
    return r / np.pi * 180


def transform(lng, lat, alt, r0=6371000, device=-1):
    theta, phi, r = d2r(lng), d2r(lat), alt + r0
    return delegate.transform(theta, phi, r)


def dtransform(lng, lat, alt, dlng, dlat, dalt, r0=6371000, device=-1):
    theta, phi, r = d2r(lng), d2r(lat), alt + r0
    dtheta, dphi, dr = d2r(dlng), d2r(dlat), dalt

    return delegate.dtransform(theta, phi, r, dtheta, dphi, dr)


def inverse(x, y, z, r0=6371000, device=-1):
    theta, phi, r = delegate.inverse(x, y, z)
    lng, lat, alt = r2d(theta), r2d(phi), r - r0

    return lng, lat, alt


def dinverse(x, y, z, dx, dy, dz, **kwargs):
    dtheta, dphi, dr = delegate.dinverse(x, y, z, dx, dy, dz)
    dlng, dlat, dalt = r2d(dtheta), r2d(dphi), dr

    return dlng, dlat, dalt
