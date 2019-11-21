# -*- coding: utf-8 -*-

import numpy as np
import torch as th
import logging

from typing import Tuple
from torch import Tensor
from torch.nn.functional import conv2d, conv3d

default_device = -1
used_modules = []
used_environ = {}

logger = logging.getLogger('leibniz')

lanczosk2d = None
lanczosk3d = None


def get_device():
    return default_device


def set_device(ix):
    global default_device, lanczosk2d, lanczosk3d

    if default_device != -1 and th.cuda.is_available():
        if lanczosk2d is not None:
            lanczosk2d = lanczosk2d.cuda(device=ix)

        if lanczosk3d is not None:
            lanczosk3d = lanczosk3d.cuda(device=ix)

    if default_device == ix:
        return

    logger.info('reset device from %d to %d', default_device, ix)

    if _grid_ is not None:
        _grid_.set_device(ix)
    if _frame_ is not None:
        _frame_.set_device(ix)
    if _element_ is not None:
        _element_.set_device(ix)

    if default_device != -1 and ix == -1:
        if lanczosk2d is not None:
            lanczosk2d = lanczosk2d.cpu()

        if lanczosk3d is not None:
            lanczosk3d = lanczosk3d.cpu()

    default_device = ix

    for mname in used_modules:
        use(mname, **used_environ[mname])

    globals().update({
        'L': _grid_.L,
        'W': _grid_.W,
        'H': _grid_.H,
        'boundary': lambda: _grid_.boundary(),
        'mk_zero': _grid_.mk_zero,
        'mk_one': _grid_.mk_one,
        'zero': _grid_.zero,
        'one': _grid_.one,
        'random': lambda: _grid_.random(),
        'dL': _element_.dL,
        'dS': _element_.dS,
        'dV': _element_.dV,
    })


def cast(element, device=-1) -> Tensor:
    element = np.array(element, dtype=np.float64)
    tensor = th.DoubleTensor(element)
    if device != -1 and th.cuda.is_available():
        return tensor.cuda(device=device)
    else:
        return tensor


diff = None
upwindp = None
upwindm = None

_grid_ = None
_frame_ = None
_element_ = None
_ops_ = None
_v3_ = None


def clear():
    global _grid_, _frame_, _element_, _ops_, _v3_
    _grid_ = None
    _frame_ = None
    _element_ = None
    _ops_ = None
    _v3_ = None

    used_modules.clear()
    used_environ.clear()


def load(mname):
    import importlib

    if ',' in mname:
        return importlib.import_module('leibniz.core3d.basis.%s' % mname.replace(',', '_'))
    else:
        return importlib.import_module('leibniz.core3d.frame.%s' % mname)


def use(mname, **kwargs):
    if mname not in used_modules:
        used_modules.append(mname)
        used_environ[mname] = kwargs

    if ',' not in mname:
        target = load(mname)
        pname = target._name_
        glbs = globals()
        pval = target._clazz_(_grid_)
        glbs.update({
            pname: pval,
        })
    else:
        data = _grid_.data()
        delta = _grid_.delta()

        if mname == _grid_.basis:
            target = load(mname)
            pvalues = data[:, 0:1], data[:, 1:2], data[:, 2:3]
            dvalues = delta[:, 0:1], delta[:, 1:2], delta[:, 2:3]
        else:
            origin = load(_grid_.basis)
            target = load(mname)

            tval1, tval2, tval3 = origin.transform(data[:, 0:1], data[:, 1:2], data[:, 2:3], **kwargs)
            pval1, pval2, pval3 = target.inverse(tval1, tval2, tval3, **kwargs)
            mval1, mval2, mval3 = origin.dtransform(
                data[:, 0:1], data[:, 1:2], data[:, 2:3],
                delta[:, 0:1], delta[:, 1:2], delta[:, 2:3], **kwargs)
            dval1, dval2, dval3 = target.dinverse(tval1, tval2, tval3, mval1, mval2, mval3, **kwargs)

            pvalues = pval1, pval2, pval3
            dvalues = dval1, dval2, dval3

        for ix, pname in enumerate(target._params_):
            dname = 'd%s' % pname
            pval = pvalues[ix]
            dval = dvalues[ix]

            globals().update({
                pname: pval,
                dname: dval
            })


def bind(grid, **kwargs):
    if 'device' in kwargs.keys():
        device_to = kwargs['device']
    else:
        device_to = default_device

    logger.info('device_to: %d', device_to)

    import leibniz.core3d.vec3 as v3
    import leibniz.core3d.diffr.regular3 as d

    from leibniz.core3d.element import Elements
    from leibniz.core3d.ops import LocalOps

    global _grid_, _frame_, _element_, _ops_, _v3_, diff, upwindm, upwindp

    _grid_ = grid
    diff = d.central5
    upwindp = d.upwind_p4
    upwindm = d.upwind_m4

    globals().update({
        'L': _grid_.L,
        'W': _grid_.W,
        'H': _grid_.H,
        'boundary': _grid_.boundary(),
        'zero': _grid_.zero,
        'one': _grid_.one,
        'random': _grid_.random(),
    })

    use(_grid_.basis, **kwargs)

    _element_ = Elements(**kwargs)
    _ops_ = LocalOps(_grid_, _element_, diff)
    _v3_ = v3

    globals().update({
        'dL': _element_.dL,
        'dS': _element_.dS,
        'dV': _element_.dV,
    })

    set_device(device_to)
    d.set_device(device_to)


def dot(x: Tuple[Tensor, Tensor, Tensor], y: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    return _v3_.dot(x, y)


def cross(x: Tuple[Tensor, Tensor, Tensor], y: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    return _v3_.cross(x, y)


def norm(v: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    return _v3_.norm(v)


def normsq(v: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    return _v3_.normsq(v)


def normalize(v: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    return _v3_.normalize(v)


def box(a: Tuple[Tensor, Tensor, Tensor], b: Tuple[Tensor, Tensor, Tensor], c: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    return _v3_.box(a, b, c)


def det(transform: Tensor) -> Tensor:
    return _v3_.det(transform)


def grad(scalar: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    return _ops_.grad(scalar)


def div(vector: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
    return _ops_.div(vector)


def curl(vector: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    return _ops_.curl(vector)


def laplacian(scalar: Tensor) -> Tensor:
    return _ops_.laplacian(scalar)


def vlaplacian(vector: Tuple[Tensor, Tensor, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
    return _ops_.vlaplacian(vector)


def adv(wind: Tuple[Tensor, Tensor, Tensor], scalar: Tensor, filter=None) -> Tensor:
    return _ops_.adv(wind, scalar, filter)


def vadv(wind: Tuple[Tensor, Tensor, Tensor], vector: Tuple[Tensor, Tensor, Tensor], filter=None) -> Tuple[Tensor, Tensor, Tensor]:
    return _ops_.vadv(wind, vector, filter)


def upwind(wind: Tuple[Tensor, Tensor, Tensor], scalar: Tensor, filter=None) -> Tensor:
    return _ops_.upwind(wind, scalar, filter, upwindp=upwindp, upwindm=upwindm)


avgk = cast([[[
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
]]]) / 27.0


def avgfilter(f):
    return conv3d(f, avgk, padding=1)


lanczosk2d = cast(np.sinc([[[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [-3, -2, -1, 0, +1, +2, +3],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]]]) * np.sinc([[[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [-3 / 3, -2 / 3, -1 / 3, 0, +1 / 3, +2 / 3, +3 / 3],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]]]))


lanczosk2d = lanczosk2d * lanczosk2d.transpose(-2, -1)


def lanczosfilter(f):
    return conv2d(f, lanczosk2d, padding=3)


lanczosk3d = cast(np.sinc([[[[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [-3, -2, -1, 0, +1, +2, +3],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]]]]) * np.sinc([[[[
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [-3 / 3, -2 / 3, -1 / 3, 0, +1 / 3, +2 / 3, +3 / 3],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
]]]]))


lanczosk3d = lanczosk3d * lanczosk3d.transpose(-2, -1) * lanczosk3d.transpose(-3, -1)


def lanczos3dfilter(f):
    return conv3d(f, lanczosk3d, padding=3)


