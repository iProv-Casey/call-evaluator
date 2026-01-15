# CallRail Recording Downloader

General-purpose CallRail call transcription and client feedback tool. It pulls recordings, transcribes them, and runs client-specific agent prompts so every stakeholder has clear visibility into what is actually happening on the phone.

## What `callrail_tool.py` Does
- Loads client credentials and system prompts (via `prompt_dir`) from `clients.yaml`; lists them with `list-clients`.
- `run`: Pulls calls from CallRail for a date range, grabs a fresh signed recording URL for each call, downloads audio with retries/fallbacks, transcribes via OpenAI Whisper, and upserts the call + transcript into Airtable (skips Airtable when `--dry-run`).
- `run-csv`: Processes only the recordings listed in a CSV (parses CallRail IDs from URLs when needed), fetches call metadata, refreshes the recording URL, downloads/transcribes, and upserts to Airtable while skipping duplicates.
- `classify-airtable`: Reads existing transcripts in Airtable, runs the per-client gatekeeper prompt (`gatekeeper.md`), and writes Gatekeeper JSON/channel/timestamp fields.
- `eval-airtable`: Reads existing transcripts in Airtable, uses the stored Gatekeeper output, runs the per-client evaluation prompt (`evaluator.md`, override with `--prompt-file`) with OpenAI Chat, and writes the evaluation JSON + executive summary (optionally stamping an evaluation timestamp). Honors `--dry-run` and `--test` limits.
- `parse-eval-json`: Parses the stored evaluation JSON into typed Airtable fields (call type, lead quality, converted status, primary issue, overall score) with optional force/dry-run/test flags.
- Common flags let you override Airtable base/table, point at an alternate clients config, and limit work to 5 items in `--test` mode.

## Why It Matters
- Gives clients an objective window into phone performance with transcripts and structured evaluations.
- Surfaces lead quality vs. front-office handling so teams know where to improve.
- Keeps feedback client-specific via per-client system prompts while sharing a unified pipeline.

## Setup
- Copy `.env.example` to `.env` and fill `CALLRAIL_API_KEY`, `OPENAI_API_KEY`, and Airtable values.
- Copy `clients.example.yaml` to `clients.yaml` and add one entry per client (CallRail `account_id` and `company_id`). Optional per-client Airtable overrides are supported. Point `prompt_dir` at each client's prompts so evaluation runs with that client's instructions.
- Install deps: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## Usage
- List configured clients: `python callrail_tool.py list-clients`.
- Run for a client and date range: `python callrail_tool.py run --client "Client Name" --start-date 2024-01-01 --end-date 2024-01-31`.
- Run only calls listed in a CSV (by recording URL): `python callrail_tool.py run-csv --client "Client Name" --csv path/to/file.csv --dry-run`.
- Classify transcripts already in Airtable (gatekeeper step): `python callrail_tool.py classify-airtable --client "Client Name" --dry-run`.
- Evaluate transcripts already in Airtable using the client's prompt: `python callrail_tool.py eval-airtable --client "Client Name" --dry-run` (use `--prompt-file` to override).
- Parse Evaluation JSON into typed Airtable columns (Call Type, Lead Quality Rating, Converted, Primary Issue, Overall Score): `python callrail_tool.py parse-eval-json --client "Client Name" --dry-run`.
- Export flattened Evaluation JSON to CSV: `python callrail_tool.py parse-eval-json --client "Client Name" --output-csv eval_export.csv`.
- Check Airtable schema for expected evaluation/parse fields: `python callrail_tool.py check-airtable-schema --client "Client Name"`.
- Flags: `--config` to point at a different clients file, `--airtable-base`/`--airtable-table` to override Airtable targets, `--dry-run` to skip Airtable writes. Add `--test` to process only 5 calls/rows. For CSV mode, use `--recording-url-column` (default `Recording URL`) and optional `--call-id-column` if the file already contains CallRail IDs.

## Airtable schema suggestion
Create a table with fields: `Client` (single line), `Call ID` (single line, make it unique), `Call Started At` (date/time), `Caller Number`, `Tracking Number`, `Duration (s)`, `Recording URL`, and `Transcript` (long text). Add `Gatekeeper JSON`, `Gatekeeper Channel`, `Gatekeeper Timestamp`, plus the evaluation fields listed below.

## Notes
- The script upserts by `Call ID` to avoid duplicates.
- CSV mode parses the CallRail ID from the recording URL when a `Call ID` column is absent.
- CallRail recording URLs are fetched with the same API token; if your org uses scoped keys per company, set them in the environment before running.
- Whisper usage can incur cost; consider `--dry-run` first to validate connectivity.
- `classify-airtable` uses `gatekeeper.md` and `eval-airtable` uses `evaluator.md` from each client's `prompt_dir` in `clients.yaml` unless `--prompt-file` is provided.
- Standard prompt filenames per client: `gatekeeper.md`, `evaluator.md`, `parser.md` (eval uses `evaluator.md`).
- `eval-airtable` expects Gatekeeper output to be present (`Gatekeeper JSON` by default); run `classify-airtable` first.
- Airtable evaluation mode expects fields: `Transcript`, `Evaluation JSON`, `Evaluation Summary`, and (optional) `Evaluation Timestamp`. Adjust via flags if your column names differ.
- Parsing mode expects single-select fields matching your configured options: `Call Type`, `Lead Quality Rating`, `Converted`, `Primary Issue`, plus a numeric `Overall Score`. Use the `parse-eval-json` flags if your column names differ.
- CSV export flattens evaluation JSON keys using dot notation and includes `Airtable Record ID`, `Call ID`, and `Client` when available.
- To rerun evaluations or parsing and overwrite existing data, use `--force`.
- Set the prompt location per client in `clients.yaml` using `prompt_dir` (or override with `--prompt-file` at runtime).
- Long-running jobs stop if you close the terminal; use `nohup` or `tmux` if you need them to keep running after disconnect.

## Making Commits (GitHub)
Use these steps to commit your changes and push them to GitHub.

1) Check the working tree: `git status`.
2) Review what changed: `git diff` (or `git diff --staged` after staging).
3) Stage the files you want: `git add README.md` (or `git add -A` to stage everything).
4) Commit with a clear message: `git commit -m "Add README commit steps"`.
5) Pull remote updates (optional but recommended): `git pull --rebase`.
6) Push to GitHub: `git push`.

# Running On Mac in VS Code

# Running On Mac in VS Code

Follow these steps from the project root directory in the VS Code terminal.

1) Verify Python 3 is installed: python3 --version
2) Create a virtual environment: python3 -m venv .venv
3) Activate the virtual environment: source .venv/bin/activate
4) Install dependencies: pip install -r requirements.txt
5) Load `.env` into your shell (one-liner): set -a; source .env; set +a
6) Set required environment variables:
   - export AIRTABLE_API_KEY="your_key_here"
   - export OPENAI_API_KEY="your_key_here"
   - export CALLRAIL_API_KEY="your_key_here"
7) Run the script: python callrail_tool.py list-clients
8) Update `clients.yaml` with the client prompt path: set `prompt_dir` for the client entry.
9) Test `classify-airtable` on 5 records: python callrail_tool.py classify-airtable --client "Client Name" --test
10) Test `eval-airtable` on 5 records and overwrite existing data: python callrail_tool.py eval-airtable --client "Client Name" --test --force
