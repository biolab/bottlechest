"Generate all pyx files from templates"

from bottleneck.src.template.func.func import funcpyx

def makepyx():
    funcpyx(bits=32)
    funcpyx(bits=64)
