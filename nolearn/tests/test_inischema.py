SAMPLE_SCHEMA = """
[first]
value1 = int
value2 = string
value3 = float
value4 = listofstrings
value5 = listofints

[second]
value1 = string
value2 = int
"""

SAMPLE_CONFIGURATION = """
[first]
value1 = 3
value2 = Three
value3 = 3.0
value4 = Three Drei
value5 = 3 3

[second]
value1 =
    a few line breaks

    are no problem

    neither is a missing value2
"""


def test_parse_config():
    from ..console import parse_config
    result = parse_config(SAMPLE_SCHEMA, SAMPLE_CONFIGURATION)
    assert result['first']['value1'] == 3
    assert result['first']['value2'] == u'Three'
    assert result['first']['value3'] == 3.0
    assert result['first']['value4'] == [u'Three', u'Drei']
    assert result['first']['value5'] == [3, 3]
    assert result['second']['value1'] == (
        u'a few line breaks\nare no problem\nneither is a missing value2')
    assert 'value2' not in result['second']
