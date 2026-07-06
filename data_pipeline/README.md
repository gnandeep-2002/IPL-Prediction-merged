# data_pipeline/ -- raw Cricsheet-to-Excel conversion (reference only)

This is the `ipl-scraper` project's pipeline for turning raw Cricsheet
match JSON into a single Excel workbook. It is kept here purely as
**provenance/reference material** for where the historical ball-by-ball
data ultimately comes from -- it is **not part of the default pipeline**
and `run_all.py` never touches this folder.

## This is NOT the same file as `data/raw/ipl_data.xlsx`

**Important -- do not confuse the two:**

- `data_pipeline/`'s output is a **raw** conversion of Cricsheet JSON:
  27 columns, 1-indexed overs, all seasons through 2026, no toss/result/
  match-outcome columns, no derived ML features (`team_runs`,
  `team_wicket`, `runs_target`, etc.).
- `../data/raw/ipl_data.xlsx` -- the file the rest of this project
  actually reads -- has been separately built/tweaked to fit what the
  models need (37 columns, 0-indexed overs, filtered to standard
  completed matches through 2025, with toss/result/derived-feature
  columns already joined in).

**Running this pipeline does not regenerate or update
`data/raw/ipl_data.xlsx`.** The two files serve different purposes and
are not interchangeable.

See `DATASET_STRUCTURE.md` in this folder for the full raw-column
reference.

## How to run

`cricsheet_raw/` (1,243 raw match JSON files, ~103MB) is not committed to
this repo -- if it isn't already present in this folder, download it
first:

```bash
curl -L https://cricsheet.org/downloads/ipl_json.zip -o ipl_json.zip
unzip ipl_json.zip -d cricsheet_raw
```

Then, from inside this directory:

```bash
cd data_pipeline
python3 convert_to_excel.py   # cricsheet_raw/*.json -> ./ipl_data.xlsx

mkdir -p output
mv ipl_data.xlsx output/ipl_data.xlsx
```

The result lands at `data_pipeline/output/ipl_data.xlsx` -- a separate
path, never `data/raw/ipl_data.xlsx`.
