test:
	uv run pytest -p no:platformdirs -n auto

jupyterlab:
	uv run jupyter-lab --no-browser
