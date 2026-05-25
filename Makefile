# Convenience targets for the semantic-kv-control-plane repo.

.PHONY: test lint bench docs clean

test:
	python -m pytest --tb=short

lint:
	python -m ruff check

bench:
	python benchmarks/run_all.py

docs:
	python scripts/generate_paper_figures.py
	python scripts/generate_blog_assets.py

clean:
	python -c "from pathlib import Path; import shutil; [shutil.rmtree(path, ignore_errors=True) for path in [Path('outputs/plots'), Path('outputs/paper_figures'), Path('outputs/blog')]]"
