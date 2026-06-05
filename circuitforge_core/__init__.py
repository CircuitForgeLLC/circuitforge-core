from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("circuitforge-core")
except PackageNotFoundError:
    __version__ = "dev"  # running from source without an editable install

try:
    from circuitforge_core.community import CommunityDB, CommunityPost, SharedStore
    __all__ = ["CommunityDB", "CommunityPost", "SharedStore"]
except ImportError:
    # psycopg2 not installed — install with: pip install circuitforge-core[community]
    pass
