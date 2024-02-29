class DummyStrategy:
    """This class provides a context manager that is compliant with the `tf.distribute` interface
    and in particular can be used interchangeably with `tf.distribute.MirroredStrategy`,
    but does not actually offer any parallelization.
    It is used to unify parallelized and non-parallelized training and testing
    into a single workflow.

    It is mainly used as:
    >>> with DummyStrategy.scope():
    ...   ...
    """

    class _ContextManager:
        """Dummy context manager, it does nothing."""

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

    @classmethod
    def scope(cls):
        return cls._ContextManager()
