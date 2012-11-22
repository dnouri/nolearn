"""The console module contains the :class:`Command` class that's
useful for building command-line scripts.

Consider a function `myfunc` that you want to call directly from the
command-line, but you want to avoid writing glue that deals with
argument parsing, converting those arguments to Python types and
passing them to other functions.  Here's how `myfunc` could look like:

.. code-block:: python

    def myfunc(a_string, a_list):
        print a_string in a_list

`myfunc` takes two arguments, one is expeced to be a string, the other
one a list.

Let's use :class:`Command` to build a console script:

.. code-block:: python

    from nolearn.console import Command

    __doc__ = '''
    Usage:
      myprogram myfunc <config_file> [options]
    '''

    schema = '''
    [myfunc]
    a_string = string
    a_list = listofstrings
    '''

    class Main(Command):
        __doc__ = __doc__
        schema = schema
        funcs = [myfunc]

    main = Main()

Note how we define a `schema` that has a definition of `myfunc`'s
arguments and their types.  See :mod:`nolearn.inischema` for more
details on that.

We can then include this `main` function in our `setup.py` to get a
console script:

.. code-block:: python

    setup(
        name='myprogram',
        # ...
        entry_points='''
        [console_scripts]
        myprogram = myprogram.mymodule.main
        ''',
        )

With this in place, you can now call the `myprogram` script like so:

.. code-block:: bash

    $ myprogram myfunc args.ini

Where `args.ini` might look like:

.. code-block:: ini

    [myfunc]
    a_string = needle
    a_list = haystack haystack needle haystack haystack

These constitute the two named arguments that will be passed into
`myfunc`.  Passing of values is always done through `.ini` files.

You may also call your script with a `--profile=<fn>` option, which
you can use to profile your program using Python's standard
:mod:`cProfile` module.

A `--pdb` option is also available which allows you to automatically
enter post-mortem debugging when your script exits abnormally.
"""

import cProfile
import pdb
import os
import sys
import traceback

import docopt

from .inischema import parse_config

DEFAULT_OPTIONS = """
Options:
  -h --help           Show this screen
  --pdb               Do post mortem debugging on errors
  --profile=<fn>      Save a profile to <fn>
"""


class Command(object):
    __doc__ = None
    schema = None
    funcs = []

    def __init__(self, **kwargs):
        vars(self).update(kwargs)

    def doc(self):
        doc = self.__doc__
        if 'Options:' not in doc:
            doc = doc + DEFAULT_OPTIONS
        return doc

    def __call__(self, argv=sys.argv):
        doc = self.doc()
        arguments = docopt.docopt(doc, argv=argv[1:])
        self.arguments = arguments

        for func in self.funcs:
            if arguments[func.__name__]:
                break
        else:  # pragma: no cover
            raise KeyError("No function found to call.")

        with open(arguments['<config_file>']) as config_file:
            self.config = parse_config(self.schema, config_file.read())

        env = self.config.get('env', {})
        for key, value in env.items():
            os.environ[key.upper()] = value

        kwargs = self.config.get(func.__name__, {})

        # If profiling, wrap the function with another one that does the
        # profiling:
        if arguments.get('--profile'):
            func_ = func

            def prof(**kwargs):
                cProfile.runctx(
                    'func(**kwargs)',
                    globals(),
                    {'func': func_, 'kwargs': kwargs},
                    filename=arguments['--profile'],
                    )
            func = prof

        # If debugging, call pdb.post_mortem() in the except clause:
        try:
            func(**kwargs)
        except:
            if arguments.get('--pdb'):
                traceback.print_exc()
                pdb.post_mortem(sys.exc_traceback)
            else:  # pragma: no cover
                raise
