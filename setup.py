#!/usr/bin/env python

import os
from setuptools import setup
from setuptools.extension import Extension
import numpy as np


CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering"]

# Description
description = "Fast NumPy array functions written in Cython"
fid = open('README.rst', 'r')
long_description = fid.read()
fid.close()
idx = max(0, long_description.find("Bottlechest is a collection"))
long_description = long_description[idx:]

# Get bottlechest version
ver_file = os.path.join('bottlechest', 'version.py')
fid = open(ver_file, 'r')
VER = fid.read()
fid.close()
VER = VER.split("= ")
VER = VER[1].strip()
VER = VER.strip("\"")
VER = VER.split('.')

NAME                = 'Bottlechest'
MAINTAINER          = "Bioinformatics Laboratory, FRI UL"
MAINTAINER_EMAIL    = "contact@orange.biolab.si"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "https://github.com/biolab/bottlechest"
DOWNLOAD_URL        = "https://github.com/biolab/bottlechest"
LICENSE             = "Simplified BSD"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Bioinformatics Laboratory, FRI UL"
AUTHOR_EMAIL        = "contact@orange.biolab.si"
PLATFORMS           = "OS Independent"
MAJOR               = VER[0]
MINOR               = VER[1]
MICRO               = VER[2]
ISRELEASED          = False
VERSION             = '%s.%s.%s' % (MAJOR, MINOR, MICRO)
PACKAGES            = ["bottlechest",
                       "bottlechest/slow",
                       "bottlechest/tests",
                       "bottlechest/benchmark",
                       "bottlechest/src",
                       "bottlechest/src/func",
                       "bottlechest/src/template",
                       "bottlechest/src/template/func"
                      ]
PACKAGE_DATA        = {'bottlechest': ['LICENSE']}
REQUIRES            = ["numpy"]


# Is the default numpy int 32 or 64 bits?
if np.int_ == np.int32:
    bits = '32'
elif np.int_ == np.int64:
    bits = '64'
else:
    raise ValueError("Your OS does not appear to be 32 or 64 bits.")

cfiles = [ a[:-2] for a in  os.listdir("bottlechest/src/func/%sbit/" % bits) \
    if a.endswith(".c") ]

extensions = [ Extension(cf,
               sources=["bottlechest/src/func/%sbit/%s.c" % (bits, cf)],
               include_dirs=[np.get_include()]) for cf in cfiles ]

setup(name=NAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      classifiers=CLASSIFIERS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      platforms=PLATFORMS,
      version=VERSION,
      packages=PACKAGES,
      package_data=PACKAGE_DATA,
      requires=REQUIRES,
      ext_package='bottlechest',
      ext_modules=extensions
      )
