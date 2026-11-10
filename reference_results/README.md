# Archived ULB aggregates

These files are the public ULB subset of the experiment archives, not results
from the smoke test or from a new full retraining.

- `test_metrics.csv`: three seeds and their ensemble for five arms.
- `test_ensemble_metrics.csv`: the ensemble rows for all five arms.
- `paired_intervals.csv`: the 2,000-draw ULB loss contrasts with frozen thresholds.
- `development_four_cell.csv`: original out-of-fold calibration diagnostics for
  the four gate/loss combinations, before adding the weighted BCE control.

The two gate-on arms come from the official-backbone factorial archive. The
gate-off BCE, NP-TRBR and weighted BCE arms and paired intervals come from the
loss-control archive. These results use the same public split and seeds.
Neither row-level predictions nor checkpoint files are distributed.
