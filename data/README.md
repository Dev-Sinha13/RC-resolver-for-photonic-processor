# Data

The synthetic Mackey–Glass benchmark is generated in code.

The real-data experiment uses the UCI Air Quality dataset:

- Saverio De Vito, *Air Quality*, UCI Machine Learning Repository (2008).
- DOI: `10.24432/C59K5F`
- License: CC BY 4.0
- Sampling: hourly averages recorded from March 2004 to February 2005.
- Default signal: `PT08.S1(CO)` metal-oxide sensor response.
- Missing marker: `-200`; the loader selects the longest contiguous valid
  hourly run.
- Splits: chronological 60/20/20.
- Scaling: mean and standard deviation from the training split only.

Downloaded files live under `data/raw/` and are intentionally ignored by Git.
