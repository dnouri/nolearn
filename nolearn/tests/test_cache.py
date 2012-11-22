from mock import patch


def test_cached(tmpdir):
    from ..cache import cached

    called = []

    @cached(cache_path=str(tmpdir))
    def add(one, two):
        called.append([one, two])
        return one + two

    assert add(2, 3) == 5
    assert len(called) == 1
    assert add(2, 3) == 5
    assert len(called) == 1
    assert add(2, 4) == 6
    assert len(called) == 2


def test_cache_with_cache_key(tmpdir):
    from ..cache import cached

    called = []

    def cache_key(one, two):
        return one

    @cached(cache_key=cache_key, cache_path=str(tmpdir))
    def add(one, two):
        called.append([one, two])
        return one + two

    assert add(2, 3) == 5
    assert len(called) == 1
    assert add(2, 3) == 5
    assert len(called) == 1
    assert add(2, 4) == 5
    assert len(called) == 1


def test_cache_systemerror(tmpdir):
    from ..cache import cached

    @cached(cache_path=str(tmpdir))
    def add(one, two):
        return one + two

    with patch('nolearn.cache.cPickle.dump') as dump:
        dump.side_effect = SystemError()
        assert add(2, 3) == 5
