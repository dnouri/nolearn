import sys

PY2 = sys.version_info[0] == 2

if PY2:
    from ConfigParser import ConfigParser
    from StringIO import StringIO
    import cPickle as pickle
else:
    from configparser import ConfigParser
    from io import StringIO
    import pickle as pickle
