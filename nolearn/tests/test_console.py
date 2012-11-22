import os
from tempfile import NamedTemporaryFile

from mock import patch

from test_inischema import SAMPLE_CONFIGURATION
from test_inischema import SAMPLE_SCHEMA


SAMPLE_CONFIGURATION += """
[env]
somekey = somevalue
"""


class TestCommand(object):

    with NamedTemporaryFile(delete=False) as config_file:
        config_file.write(SAMPLE_CONFIGURATION)

    def some_filename(self):
        with NamedTemporaryFile(delete=False) as some_file:
            return some_file.name

    def test_simple(self):
        from ..console import Command

        called = []

        def second(value1, value2=None):
            called.append((value1, value2))

        class MyCommand(Command):
            __doc__ = """
            Usage:
              script second <config_file>
            """
            schema = SAMPLE_SCHEMA
            funcs = [second]

        argv = ['script', 'second', self.config_file.name]
        MyCommand()(argv)
        assert len(called) == 1
        assert called[0][0].startswith('a few line breaks')
        assert called[0][1] is None
        assert os.environ['SOMEKEY'] == 'somevalue'

    def test_profiler(self):
        from ..console import Command

        called = []

        def second(value1, value2=None):
            called.append((value1, value2))

        class MyCommand(Command):
            __doc__ = """
            Usage:
              script second <config_file> [--profile=<file>]
            """
            schema = SAMPLE_SCHEMA
            funcs = [second]

        profile_filename = self.some_filename()
        argv = ['script', 'second', self.config_file.name,
                '--profile', profile_filename]
        MyCommand()(argv)
        assert len(called) == 1

        with open(profile_filename) as f:
            assert(len(f.read()) > 1)

    @patch('nolearn.console.pdb.post_mortem')
    @patch('nolearn.console.traceback.print_exc')
    def test_pdb(self, print_exc, post_mortem):
        from ..console import Command

        called = []

        def second(value1, value2=None):
            called.append((value1, value2))
            raise ValueError()

        class MyCommand(Command):
            __doc__ = """
            Usage:
              script second <config_file> [--pdb]
            """
            schema = SAMPLE_SCHEMA
            funcs = [second]

        argv = ['script', 'second', self.config_file.name, '--pdb']
        MyCommand()(argv)
        assert len(called) == 1
