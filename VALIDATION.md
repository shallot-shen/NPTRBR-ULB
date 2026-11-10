# Release validation

The public-data preparation command successfully downloaded OpenML 1597, kept
the original row order and generated the manuscript partition counts (284,807
rows, 492 positive labels; 56,961 rows and 75 positives in test). Data files
and manifests produced by that check are not part of the release.

The synthetic CPU smoke test covered all five arms, two seeds, completed-run
reuse, restart after an interrupted epoch, the official-backbone identity check,
calibration, 60 confusion matrices and paired bootstrap output. Four fast
protocol tests checked partition fingerprints, seed alignment, threshold caps
and frozen control settings. The table builder was also run against the five
archived public ensemble rows per operating rule.

Validation environment: Python 3.12, PyTorch 2.5.1+cpu, NumPy 2.3.5, pandas
2.2.3, scikit-learn 1.8.0, SciPy 1.17.0, TabM 0.0.3 and
rtdl-num-embeddings 0.0.12. This is not a reconstruction of the original GPU
environment. Full real-data training, CUDA and the optional baseline suites
were not rerun for packaging. The archived reference values are therefore
reported separately from fresh validation.
