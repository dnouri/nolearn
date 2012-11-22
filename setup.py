import os

from setuptools import setup, find_packages

version = '0.1dev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
except IOError:
    README = ''

install_requires = [
    'docopt',
    'scikit-learn',
    ]

maybe_requires = [
    'numpy',
    'scipy',
    ]

if os.environ.get('SETUPTOOLS_INSTALL_ALL_DEPS', '1').lower() in ('1', 'y'):
    install_requires = maybe_requires + install_requires


tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    ]

docs_require = [
    'Sphinx',
    ]

setup(name='nolearn',
      version=version,
      description="Miscellaneous utilities for machine learning.",
      long_description=README,
      classifiers=[
          'Development Status :: 4 - Beta',
        ],
      keywords='',
      author='Daniel Nouri',
      author_email='daniel.nouri@gmail.com',
      url='https://github.com/dnouri/nolearn',
      license='MIT',
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      extras_require={
          'testing': tests_require,
          'docs': docs_require,
          },
      )
