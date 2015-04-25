def pytest_configure(config):
    # Make matplotlib happy when running without an X display:
    import matplotlib
    matplotlib.use('Agg')
