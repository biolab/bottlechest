"""
This file is only Used to build the .so files with "make build".

This setup.py is NOT used to install the Bottlechest package. The Bottlechest
setup.py file is bottlechest/setup.py
"""

import os
import os.path
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

# Is the default numpy int 32 or 64 bits?
if np.int_ == np.int32:
    bits = '32'
elif np.int_ == np.int64:
    bits = '64'
else:
    raise ValueError("Your OS does not appear to be 32 or 64 bits.")


cfiles = [ a[:-2] for a in os.listdir("bottlechest/src/func/%sbit/" % bits) \
    if a.endswith(".c") ]

extensions = [ Extension(cf,
               sources=["bottlechest/src/func/%sbit/%s.c" % (bits, cf)],
               include_dirs=[np.get_include()]) for cf in cfiles ]

setup(
  name = 'func',
  ext_package= "bottlechest",
  ext_modules = extensions
)

