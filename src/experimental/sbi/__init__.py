"""SBI-style training with explicit landmarks and audited inference."""


def fit(*args, **kwargs):
    from .training import fit as implementation
    return implementation(*args, **kwargs)


def load_model(*args, **kwargs):
    from .training import load_model as implementation
    return implementation(*args, **kwargs)


def describe_run(*args, **kwargs):
    from .training import describe_run as implementation
    return implementation(*args, **kwargs)
