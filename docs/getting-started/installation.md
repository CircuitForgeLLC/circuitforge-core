# Installation

circuitforge-core is distributed as an editable install from a local clone. It is not yet on PyPI.

## Prerequisites

- Python 3.11+
- A Python environment — conda or venv (see options below)
- The `circuitforge-core` repo cloned alongside your product repo

## Typical layout

```
/Library/Development/CircuitForge/
├── circuitforge-core/   ← this repo
├── kiwi/
├── peregrine/
├── snipe/
└── ...
```

## Install

### Option A: conda (dev machines)

The CircuitForge conda environment is named `cf`:

```bash
# From inside a product repo, assuming circuitforge-core is a sibling
conda run -n cf pip install -e ../circuitforge-core

# Or activate first, then install
conda activate cf
pip install -e ../circuitforge-core
```

### Option B: venv (server and beta-host deployments)

For hosts that don't use conda (CI runners, beta VMs, Xander's orchard nodes):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e /path/to/circuitforge-core
```

Or if cf-core is a sibling directory of the product:

```bash
pip install -e ../circuitforge-core
```

The editable install means changes to circuitforge-core source are reflected immediately in all products without reinstalling. Only restart the product's process after changes (or Docker container if running in Docker).

## Verify

```python
import circuitforge_core
print(circuitforge_core.__version__)  # e.g. 0.21.0
```

## Inside Docker

Product Dockerfiles copy or mount both the product source and cf-core:

```dockerfile
# Copy cf-core alongside product source
COPY --from=build /circuitforge-core /circuitforge-core
RUN pip install -e /circuitforge-core
```

The `compose.yml` for each product typically bind-mounts both directories in dev mode so live edits propagate without rebuilding the image.

## Upgrading

cf-core follows semantic versioning. Since it's an editable install, `git pull` in the cf-core repo is sufficient — no reinstall needed for pure Python changes.

For schema changes (new migrations) or new module dependencies, check the CHANGELOG for any additional steps.
