from __future__ import absolute_import, print_function, division


class DuplicateKeyError(Exception):

    def __init__(self, key):
        self.key = key

    def __str__(self):
        return 'duplicate key: %r' % (self.key,)


class FieldSelectionError(Exception):

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return 'selection is not a field or valid field index: %r' % (self.value,)


class ArgumentError(Exception):

    def __init__(self, message):
        self.message = message

    def __str__(self):
        return 'argument error: %s' % self.message


class CalibrationError(ArgumentError):
    """Raised when weight calibration cannot produce a valid result.

    The :attr:`reason` attribute is one of:

    * ``'infeasible'`` - the margin targets cannot be met within the
      multiplier bounds (or the structural categories available in the
      sample);
    * ``'iteration_limit'`` - the solver ran through `max_iter`
      iterations without reaching the requested tolerance;
    * ``'numerical_failure'`` - the numerical procedure broke down
      (e.g. the optional SciPy solver reported an error).

    """

    def __init__(self, message, reason):
        super(CalibrationError, self).__init__(message)
        self.reason = reason

    reason = None
    """One of ``'infeasible'``, ``'iteration_limit'`` or
    ``'numerical_failure'``; see :class:`petl.errors.CalibrationError`."""

    def __str__(self):
        return 'calibration error (%s): %s' % (self.reason, self.message)
