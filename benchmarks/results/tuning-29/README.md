# Preliminary five-dataset study

This separate study completed 512 trials on credit-g, diabetes, maternal health
risk, airfoil self-noise, and concrete strength. Its Pareto configurations were
reevaluated as warm starts for the expanded 12-dataset study. Its objectives are
not pooled with the expanded objectives, and its selected configuration is not
the adopted package default.

`trials.jsonl.gz` preserves every trial's parameters, objective values, and
dataset/fold records. `manifest.json` and `search-sources.zip` preserve provenance.
The local SQLite database and redundant CSV exports are retained but ignored by
Git. The preliminary `sensitivity.json.gz` is partial: that analysis was stopped
when the user requested the larger panel. Use the expanded report and complete
sensitivity results for recommendations.
