import sys

import numpy as np


def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


class ChunkedTransform(object):
    verbose = 0

    def transform(self, X):
        features = None
        for chunk in chunks(X, self.batch_size):
            if features is not None:
                features = np.vstack(
                    [features, self._compute_features(chunk)])
            else:
                features = self._compute_features(chunk)
            if self.verbose:
                sys.stdout.write(
                    "\r[%s] %d%%" % (
                        self.__class__.__name__,
                        100. * len(features) / len(X),
                        ))
                sys.stdout.flush()
        if self.verbose:
            sys.stdout.write('\n')
        return features
