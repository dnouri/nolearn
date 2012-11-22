""":mod:`inischema` allows the definition of schemas for `.ini`
configuration files.

Consider this sample schema:

.. doctest::

    >>> schema = '''
    ... [first]
    ... value1 = int
    ... value2 = string
    ... value3 = float
    ... value4 = listofstrings
    ... value5 = listofints
    ...
    ... [second]
    ... value1 = string
    ... value2 = int
    ... '''

This schema defines the sections, names and types of values expected
in a schema file.

Using a concrete configuration, we can then use the schema to extract
values:

.. doctest::

    >>> config = '''
    ... [first]
    ... value1 = 2
    ... value2 = three
    ... value3 = 4.4
    ... value4 = five six seven
    ... value5 = 8 9
    ...
    ... [second]
    ... value1 = ten
    ... value2 = 100
    ... value3 = what?
    ... '''

    >>> result = parse_config(schema, config)
    >>> from pprint import pprint
    >>> pprint(result)
    {'first': {'value1': 2,
               'value2': 'three',
               'value3': 4.4,
               'value4': ['five', 'six', 'seven'],
               'value5': [8, 9]},
     'second': {'value1': 'ten', 'value2': 100, 'value3': 'what?'}}

Values in the config file that are not in the schema are assumed to be
strings.

This module is used in :mod:`nolearn.console` to allow for convenient
passing of values from `.ini` files as function arguments for command
line scripts.
"""

from ConfigParser import ConfigParser
from StringIO import StringIO


def string(value):
    return value.strip()


def listofstrings(value):
    return [string(v) for v in value.split()]


def listofints(value):
    return [int(v) for v in value.split()]


converters = {
    'int': int,
    'string': string,
    'float': float,
    'listofstrings': listofstrings,
    'listofints': listofints,
    }


def parse_config(schema, config):
    schemaparser = ConfigParser()
    schemaparser.readfp(StringIO(schema))
    cfgparser = ConfigParser()
    cfgparser.readfp(StringIO(config))

    result = {}
    for section in cfgparser.sections():
        result_section = {}
        schema = {}
        if section in schemaparser.sections():
            schema = dict(schemaparser.items(section))
        for key, value in cfgparser.items(section):
            converter = converters[schema.get(key, 'string')]
            result_section[key] = converter(value)
        result[section] = result_section
    return result
