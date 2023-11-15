"""gtn_copliot package."""
from importlib import metadata

from gtn_copliot.chain import get_graphql_chain

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""

__all__ = [__version__, "get_graphql_chain"]
