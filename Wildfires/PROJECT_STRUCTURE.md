# Project structure & migration plan

This document explains the proposed directory structure and the steps to migrate and operationalize the repository.

## Goals
- Keep raw inputs immutable (data/raw).
- Store deterministic pipeline outputs in data/processed.
- Refactor long notebooks into small, testable scripts under src/.
- Keep notebooks for exploration and demonstrations under notebooks/.
- Store models and deliverables under models_output/ and reports/.

## Mapping (existing -> target)
- 01-webscraping-FINAL.ipynb -> notebooks/01-data-acquisition.ipynb
  - Extract API calls into src/data/fetch_weather.py and src/data/fetch_fire_data.py
  - Save raw JSON/CSV to data/raw/
- 02-preprocessing-FINAL.ipynb -> notebooks/02-preprocessing.ipynb
  - Move transformation functions to src/preprocessing/clean_fire.py and clean_weather.py
  - Save processed CSVs to data/processed/
- 03-resampling.ipynb -> notebooks/03-resampling.ipynb
  - Implement resampling as a function in src/preprocessing/combine.py
- 05-model-classification-FINAL.ipynb -> notebooks/05-model-classification.ipynb
  - Move training logic to src/models/train_classifiers.py and train_nn.py
  - Save model artifacts to models_output/

## Suggested immediate steps (small checklist)
1. Create the directories listed in PROPOSED_TREE.txt.
2. Move CSV / HTML assets into data/raw and reports as shown.
3. Add requirements.txt and optionally environment.yml.
4. Refactor notebook cells into modular scripts inside src/ (follow tests-driven approach).
5. Add basic tests under tests/ for ingestion/cleaning functions.
6. Add a Makefile with tasks for setup, run-pipeline, train, and test.
7. Run pipeline end-to-end and store outputs in data/processed and models_output/.

## Suggested priority tasks
- Task A: Add src/utils/config.py with path constants and load them in scripts.
- Task B: Implement fetch scripts and confirm data/raw contains expected files.
- Task C: Implement cleaning and a single CLI entrypoint (e.g., `python src/preprocessing/combine.py --run`) to create combined.csv.
- Task D: Implement a train script that consumes data/processed/combined.csv and outputs a model in models_output/.
- Task E: Move visualization HTML to reports/ and ensure it renders relative-to-repo static path.

## Notes on reproducibility
- Pin package versions in requirements.txt / environment.yml.
- Keep raw data files in `data/raw` and exclude processed artifacts from version control if large (use Git LFS or .gitignore rules as appropriate).
- Commit small, tested refactors rather than moving everything in a single commit — this will make debugging easier.
