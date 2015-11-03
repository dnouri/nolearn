import os

from setuptools import setup, find_packages

version = '0.6adev'

here = os.path.abspath(os.path.dirname(__file__))
try:
    README = open(os.path.join(here, 'README.rst')).read()
    CHANGES = open(os.path.join(here, 'CHANGES.txt')).read()
except IOError:
    README = CHANGES = ''

install_requires = [
    'joblib',
    'matplotlib',
    'scikit-learn',
    'tabulate',
    'Lasagne',
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
      description="nolearn contains a number of wrappers and abstractions "
      "around existing neural network libraries, most notably Lasagne, "
      "along with a few machine learning utility modules.  "
      "All code is written to be compatible with scikit-learn.",
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
          'testing': tests_require,
          'docs': docs_require,
          },
      )

