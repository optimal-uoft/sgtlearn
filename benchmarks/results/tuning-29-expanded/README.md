# Expanded 12-dataset study

The [analysis report](../../../docs/research/2026-10-26-tuning-defaults.md) explains
the selected common defaults, exact protocol, test outcomes, tuning priorities,
and limitations.

- `selection.json`: fixed Pareto compromise, selected before final tests.
- `manifest.json`, `search-sources.zip`: versions, hashes, and exact search sources.
- `trials.jsonl.gz`: all 512 trials, including parameters and every dataset/fold.
- `pareto.png`: classification–regression tradeoffs; the script also generates a local SVG.
- `dataset_best.json`: individually best observed configuration per dataset.
- `importance.json`: fANOVA estimates, conditional on this finite search.
- `sensitivity.json.gz`: controlled validation sweeps around the shared configuration.
- `tuning_ranges.json`: promising tested values and local effects per dataset.
- `confirmation.json`: reserved outer tests and two datasets excluded from tuning.
- `pilot.json`: runtime and baseline measurements, including the rich pilot timeout.

Local data caches contain exact split row indices. The SQLite database, data,
process state, logs, and redundant uncompressed CSV files are ignored by Git.
The compressed trial archive contains the complete measured outcomes.
The analysis script uses a local uncompressed `sensitivity.json` cache; decompress
the archive into that filename to reuse its completed sweeps in another checkout.
