import os
import codecs

from setuptools import setup, find_packages

version = '0.6.1'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = codecs.open(os.path.join(here, 'README.rst'),
                         encoding='utf-8').read()
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    README = CHANGES = ''


install_requires = [
    'joblib',
    'scikit-learn',
    'tabulate',
    'Lasagne',
    'Theano',
    ]

visualization_require = [
    'matplotlib<3.0.999',
    'pydotplus',
    'ipython<5.999'
    ]

tests_require = [
    'mock',
    'pytest',
    'pytest-cov',
    'pytest-flakes',
    'pytest-pep8',
    ]

docs_require = [
    'Sphinx<1.999',
    ]

gdbn_require = [
    "gdbn"
]

all_require = (visualization_require + tests_require + docs_require + gdbn_require)

setup(name='nolearn',
      version=version,
      description="scikit-learn compatible neural network library",
      long_description='\n\n'.join([README, CHANGES]),
      classifiers=[
          "Development Status :: 4 - Beta",
          "Programming Language :: Python :: 2.7",
          "Programming Language :: Python :: 3.4",
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
          'visualization': visualization_require,
          'testing': tests_require,
          'docs': docs_require,
          'gdbn': gdbn_require,
          'all': all_require,
          },
      )
