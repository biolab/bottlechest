"Generate all pyx files from templates"

from bottleneck.src.template.package import funcpyx

def makepyx():
    funcpyx("func", bits=32)
    funcpyx("func", bits=64)
