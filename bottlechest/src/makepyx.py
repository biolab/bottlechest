"""Generate pyx files from templates.

To generate a single file, run this script with the argument.
For example, use "python makepyx.py func.median"
"""

from bottlechest.src.template.package import (package, single)

def makepyx(): #makes all pyxfiles
    package("func", bits=32)
    package("func", bits=64)

def makepyxsingle(name):
    single(name, bits=32)
    single(name, bits=64)

if __name__ == "__main__":
    import sys
    makepyxsingle(sys.argv[1])
    
