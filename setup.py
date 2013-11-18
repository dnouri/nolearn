import os

from setuptools import setup, find_packages

version = '0.3.1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'docopt',
    'gdbn',
    'scikit-learn',
    'zope.deprecation',
    ]

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
      long_description='\n\n'.join([README, CHANGES]),
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
      dependency_links=[
        'http://danielnouri.org/python-packages/',
        ],
      )

