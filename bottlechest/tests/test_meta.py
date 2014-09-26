import imp
import inspect
import os

import bottlechest


def test_signatures_of_slow_functions():
    dir = os.path.join(bottlechest.__path__[0], 'src', 'template', 'func')
    for file in os.listdir(dir):
        if not file.endswith('.py'):
            continue
        mod = imp.load_source("tmp", os.path.join(dir, file))
        if not hasattr(mod, 'slow'):
            continue
        yield check_signature, mod.slow['name'], mod.slow['signature']


def check_signature(name, signature):
    func = getattr(bottlechest.slow, name)
    func_signature = [arg for arg in inspect.getargspec(func).args
                      if arg != 'axis']

    signature = [x.strip() for x in signature.split(',')]
    assert signature == func_signature, \
        "Signature of the slow function %s(%s) differes from " \
        "the signature in template (%s)" % (name, func_signature, signature)


if __name__ == "__main__":
    import nose

    nose.run(defaultTest='test_meta', argv=['--with-doctest', '-vv'])
