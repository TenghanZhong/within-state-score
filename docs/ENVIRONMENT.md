# Environment Notes

The repository uses a small Python stack:

```text
numpy
pandas
scipy
torch
matplotlib
pyarrow
```

The code was written for Python 3.10 or newer. Install the portable dependency
set from the repository root with:

```bash
pip install -r requirements.txt
```

The included tables and figures can be inspected without installing the Python
dependencies. The dependencies are needed to rebuild figures, rerun the frozen
500-path evaluation, or regenerate the asset-level mechanism diagnostic.

PyTorch wheels are platform-specific. If the default `pip install torch`
command does not match the local system, install PyTorch following the official
instructions for the target CPU or GPU environment, then install the remaining
packages from `requirements.txt`.
