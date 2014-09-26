===========
Bottlechest
===========

Introduction
============

Bottlechest is a fork of bottleneck (https://github.com/kwgoodman/bottleneck), specialized for use in Orange (https://github.com/biolab/orange3).

Moving window functions, several other functions and all optimization of 3d arrays are removed to reduce the size of the library. New functions are added as needed.

===================== =======================================================
NumPy/SciPy           ``median, nanmedian, rankdata, ss, nansum, nanmin,
                      nanmax, nanmean, nanstd, nanargmin, nanargmax`` 
Functions             ``nanrankdata, nanvar, replace, nn, anynan, allnan,
                      nanequal``
===================== =======================================================

For other documentation, including a simple example and comprehensive set of benchmarks, refer to the original project.

License
=======

Bottlechest is distributed under a Simplified BSD license. Parts of Bottleneck, NumPy,
Scipy, numpydoc and bottleneck, all of which have BSD licenses, are included in
Bottlechest. See the LICENSE file, which is distributed with Bottlechest, for
details.


Install
=======

Requirements:

======================== ====================================================
Bottlechest              Python 2.6, 2.7, 3.2; NumPy 1.8
Unit tests               nose
Compile                  gcc or MinGW
Optional                 SciPy 0.8, 0.9, 0.10 (portions of benchmark)
======================== ====================================================


