import warnings

from nolearn import decaf


warnings.warn("""
The nolearn.convnet module will be renamed to nolearn.decaf in nolearn
0.6.  The ConvNetFeatures class can now be found in
nolearn.decaf.ConvNetFeatures.
""")


ConvNetFeatures = decaf.ConvNetFeatures
