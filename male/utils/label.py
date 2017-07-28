from sklearn.preprocessing import LabelEncoder
from sklearn.utils.fixes import np_version
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import column_or_1d
import numpy as np


def _check_numpy_unicode_bug(labels):
    """Check that user is not subject to an old numpy bug

    Fixed in master before 1.7.0:

      https://github.com/numpy/numpy/pull/243

    """
    if np_version[:3] < (1, 7, 0) and labels.dtype.kind == 'U':
        raise RuntimeError("NumPy < 1.7.0 does not implement searchsorted"
                           " on unicode data correctly. Please upgrade"
                           " NumPy to use LabelEncoder with unicode inputs.")


class LabelEncoderDict(LabelEncoder):
    def __init__(self):
        self.encoded_label = None
        self.original_label = None
        self.encode_dict = None
        self.decode_dict = None

    def fit(self, y):
        """Fit label encoder

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : returns an instance of self.
        """
        super(LabelEncoderDict, self).fit(y)
        self.encoded_label = np.arange(len(self.classes_))
        self.original_label = self.classes_
        self.decode_dict = dict(zip(self.encoded_label, self.original_label))
        self.encode_dict = dict(zip(self.original_label, self.encoded_label))
        return self

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """

        self.fit(y)
        y_new = self.transform(y)
        return y_new

    def transform(self, y):
        """Transform labels to normalized encoding.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Target values.

        Returns
        -------
        y : array-like of shape [n_samples]
        """

        check_is_fitted(self, 'classes_')
        y = column_or_1d(y, warn=True)

        y_new = np.array([self.encode_dict[y_i] for y_i in y])
        return y_new

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        y : numpy array of shape [n_samples]
        """
        check_is_fitted(self, 'classes_')

        y_new = np.array([self.decode_dict[y_i] for y_i in y])
        return y_new
