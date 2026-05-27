# Demo Prerequisites

Scripts to set up infrastructure required by the Bedrock Smart Router demo.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup_database.py` | Creates and seeds the SQLite database used by the Text2SQL agent |
| `setup_guardrail.py` | Creates a Bedrock Guardrail with PII detection, content filtering, and denied topics |
| `setup_all.py` | Runs all prerequisite scripts in order |

## Usage

Run everything at once:

```bash
python demo/prerequisite/setup_all.py
```

Or run individually:

```bash
python demo/prerequisite/setup_database.py
python demo/prerequisite/setup_guardrail.py
```

## Generated Files

- `.guardrail_config.json` — Stores the guardrail ID and version (gitignored)

## Notes

- All scripts are idempotent — safe to run multiple times.
- The guardrail is created in `us-west-2`.
- AWS credentials with Bedrock permissions are required for the guardrail setup.
