# Aggregate Thesis Results

The perceptual evaluation contained 10 triplets. The private research snapshot
contains 571 judgments from 61 sessions: 492 judgments from 53 public sessions
and 79 judgments from 8 expert sessions. Participant names and individual
responses are not published.

Using the majority human choice for each triplet, the observed Human-Model
Agreement was 70% for MusiCNN pretrained on MSD, 70% for VGG pretrained on MSD,
60% for the multilabel contrastive Whisper variant, and 40% for the MusiCNN
MultiSignal variant.

These results are exploratory. With only 10 triplets, the uncertainty intervals
are wide, and the paired McNemar comparisons did not establish a statistically
significant difference between the evaluated models at the 0.05 level. The
percentages should not be interpreted as a definitive model ranking.

The calculation procedure, confidence intervals, paired comparisons, and
expert/public agreement analysis are documented in the sanitized notebook at
`backend/reports/hma_analysis.ipynb`. Exact recomputation requires authorized
access to the frozen private databases used for the thesis.
