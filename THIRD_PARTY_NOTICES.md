# Third-party code

`paper_suite/vendor/ma2024/ULDPFS/` contains author-derived routines from [ULDP-SL](https://github.com/Karlmyh/ULDP-SL), associated with Ma, Jia, and Yang, *Better Locally Private Sparse Estimation Given Multiple Samples Per User*, ICML 2024.

The original MIT license is preserved at `paper_suite/vendor/ma2024/LICENSE`. Upstream source references are in `source.json`; adaptation metadata is in `frozen_extraction.json`. `VERSION_LOCK.json` verifies the distributed files. Vendored code and comments are retained; text line endings are normalized to LF for portable checkouts.

Python dependencies retain their own licenses. The full Kelmarsh dataset is distributed by its publisher and is not included here.

The real-data reproduction uses *Kelmarsh wind farm data* by Charlie Plumley and Roberta Takeuchi, DOI [10.5281/zenodo.16807551](https://doi.org/10.5281/zenodo.16807551), licensed under [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/). The download and preparation scripts construct the required data inputs from the publisher record.
