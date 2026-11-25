"""Load the standalone architecture for the named experiment."""
from importlib import import_module


def build_model(variant):
    return import_module(variant).build_model()
