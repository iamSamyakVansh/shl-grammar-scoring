"""SHL grammar scoring: spoken-English grammar score regression (1-5)."""

# On Windows, importing pandas/pyarrow before torch breaks torch's DLL loading
# (WinError 1114). Import torch first so every entry point is safe.
try:
    import torch  # noqa: F401
except ImportError:
    pass
