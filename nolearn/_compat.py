import sys

PY2 = sys.version_info[0] == 2

if PY2:
    from ConfigParser import ConfigParser
    from StringIO import StringIO
    import cPickle as pickle
    import __builtin__ as builtins

    basestring = basestring

    def chain_exception(exc1, exc2):
        exec("raise exc1, None, sys.exc_info()[2]")

else:
    from configparser import ConfigParser
    from io import StringIO
    import pickle as pickle
    import builtins

    basestring = str

    def chain_exception(exc1, exc2):
        exec("raise exc1 from exc2")



