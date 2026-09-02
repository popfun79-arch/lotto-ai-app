# Lotto64 v4.5.4 — Skip Sum Transition

- Added exact skip-period frequency distribution and round-level skip-sum bands.
- Added six per-round skip values, 5/10/20-round moving averages, causal state,
  direction, and state-transition signatures.
- Added a next-round skip-pattern forecast with state+direction matching and
  state/recent fallbacks when samples are insufficient.
- Added transition-conditioned skip-sum, bucket-composition, and empirical
  hazard scores to Final Pattern (29% combined weight).
- Corrected the six-point baseline difference between GAP sums and `skip=0`
  sums in ranking and Historical Validation Ledger calculations.
- Added skip forecast hit rate and MAE fields to the historical ledger.
- Expanded Streamlit charts, metrics, tables, and tests for the new analysis.
