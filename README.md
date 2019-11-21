# Leibniz

[![Build Status](https://api.travis-ci.com/caiyunapp/leibniz.svg?branch=master)](http://travis-ci.com/caiyunapp/leibniz) 

Leibniz is a python package which provide facilities to express learnable differential equations with PyTorch


Install
--------

```bash
pip install leibniz
```


How to use
-----------

As an example we solve a very simple advection problem, a box-shaped material transported by a constant steady wind.

![moving box](https://raw.githubusercontent.com/caiyunapp/leibniz/master/advection_3d.gif)


```python
import torch as th
import leibniz as lbnz

from leibniz.core3d.gridsys.regular3 import RegularGrid
from leibniz.diffeq import odeint as odeint


def binary(tensor):
    return th.where(tensor > lbnz.zero, lbnz.one, lbnz.zero)

# setup grid system
lbnz.bind(RegularGrid(
    basis='x,y,z',
    W=51, L=151, H=51,
    east=16.0, west=1.0,
    north=6.0, south=1.0,
    upper=6.0, lower=1.0
))
lbnz.use('x,y,z') # use xyz coordinate

# giving a material field as a box 
fld = binary((lbnz.x - 8) * (9 - lbnz.x)) * \
      binary((lbnz.y - 3) * (4 - lbnz.y)) * \
      binary((lbnz.z - 3) * (4 - lbnz.z))

# construct a constant steady wind
wind = lbnz.one, lbnz.zero, lbnz.zero

# transport value by wind
def derivitive(t, clouds):
    return - lbnz.upwind(wind, clouds)

# integrate the system with rk4
pred = odeint(derivitive, fld, th.arange(0, 7, 1 / 100), method='rk4')
```

Contributors
------------

* Mingli Yuan ([Mountain](https://github.com/mountain))
* Xiang Pan ([Panpanx](https://github.com/Panpanx))

Acknowledge
-----------

We included source code with minor changes from [torchdiffeq](https://github.com/rtqichen/torchdiffeq) by Ricky Chen,
because of two purpose:
1. package torchdiffeq is not indexed by pypi
2. package torchdiffeq is very convenient and mandatory

All our contribution is based on Ricky's Neural ODE paper (NIPS 2018) and his package.

 
