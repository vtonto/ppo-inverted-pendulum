"""Execute every teaching notebook in a fresh project kernel.

Run after installing .[dev,notebook] and registering the ppo-pendulum kernel:
    python scripts/validate_notebooks.py

Saves executed outputs in place. Exercise checks remain opt-in.
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "notebooks").glob("[0-9][0-9]_*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook,
            timeout=180,
            kernel_name="ppo-pendulum",
            resources={"metadata": {"path": str(path.parent)}},
        )
        client.execute()
        nbformat.write(notebook, path)
        print(f"PASS {path.name}", flush=True)


if __name__ == "__main__":
    main()
