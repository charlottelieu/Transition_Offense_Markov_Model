# Transition Markov Model for NCAA Basketball Offense

This repository accompanies the paper:
"Evaluating the Expected Value of Transition Offense Using a Markov Chain Framework"

# Overview

This project models NCAA Division I basketball possessions as a discrete-time Markov chain.

Possessions are classified into:

- Transition (T)
- Half-Court (H)

and terminate in one of five absorbing states:

- S2 (Made 2)
- S3 (Made 3)
- TO (Turnover)
- E (Empty Possession)
- F (Free Throws)

Expected values are estimated from play-by-play data and compared across teams.

# Files

- ncaa_data_track.py: possession extraction and state classification
- bootstrap.py: confidence interval estimation
- clean_plays.csv: example processed play-by-play file

# Data Source

Play-by-play data obtained from ESPN's public API.
