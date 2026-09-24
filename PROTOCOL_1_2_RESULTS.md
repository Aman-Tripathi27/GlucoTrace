# Protocol 1.2 results: a documented negative result

Protocol 1.2 ([EVALUATION_1_2.md](EVALUATION_1_2.md)) added two release gates to
the five protocol 1.1 checks: a **source-leakage** gate and a **hidden-window
utility** gate. Six candidates, the full declared budget, were trained from
random weights and measured on validation only.

**No candidate passed all seven checks.** As declared, the protocol 1.2 test
partition was **not opened**. The released checkpoint is unchanged.

This is representation research only. None of these numbers are clinical,
diagnostic, or safety evidence.

## Candidates

All candidates used the candidate 6 recipe (40 epochs, batch size 32, seed 7,
562 source-balanced draws per epoch) on the protocol 1.2 training partition.
They differ only in the new source-invariance options in `glucofm-pretrain`.

| Candidate | Within-source negatives | Adversary weight |
|-----------|:-----------------------:|:----------------:|
| a | yes | 0 |
| b | no | 0.1 |
| c | yes | 0.1 |
| d | no | 1.0 |
| e | yes | 1.0 |
| f | yes | 3.0 |

- **Within-source negatives:** the contrastive loss compares a day only with
  other days from the same dataset, so dataset identity cannot help tell two
  days apart.
- **Adversary:** a gradient-reversal classifier tries to predict the dataset
  from the day embedding, and the encoder is trained to defeat it. The
  classifier is used in training only and is never saved.

## Validation results

The summary baseline's linear source probe reached a balanced accuracy of
`0.686`, so gate 6 required the model to stay at or below `0.786`. Summary
baseline hidden-window MAE: `8.04` mg/dL.

| Candidate | Source probe (bal. acc.) | Source margin | Hidden-window MAE | Same-person top-1 | Effective rank | Failed checks |
|---|---|---|---|---|---|---|
| a | 0.919 | 0.233 | **6.96** | 0.242 | 8.24 | source |
| b | 0.888 | 0.202 | 7.27 | 0.242 | 8.93 | source |
| c | **0.879** | **0.193** | 7.40 | 0.242 | 8.55 | source |
| d | 0.919 | 0.233 | 7.69 | 0.258 | 8.36 | source |
| e | 0.888 | 0.202 | 7.42 | 0.212 | 7.43 | source, rank |
| f | 0.888 | 0.202 | 8.42 | 0.273 | 1.65 | source, rank, utility, cadence |

Same-person top-1 chance level is `0.035`; the summary baseline reaches
`0.227`. Complete reports: `evaluations/protocol-1.2-candidate-*-validation.json`.

## What we learned

1. **The embedding is useful.** Every candidate except f predicts the mean
   glucose of the hidden final six hours better than eleven summary
   statistics, with 4 to 13% lower error. The released candidate 6 does the same
   on validation (7.71 vs 8.22 mg/dL, fitted on protocol 1.1 training data).
2. **Source leakage came down but did not come down far enough.** Adversarial and
   within-source training reduced the linear source probe from about 0.98
   (candidate 6) to 0.88. Pushing harder (e, f) collapsed the embedding
   instead of removing more source information.
3. **The likely cause is how the data is windowed, not the model.** In the
   protocol 1.2 training partition, 278 of 281 Colas days start at midnight,
   while BIG IDEAs days start at many clock times. The model receives circular
   time-of-day and pools six-hour segments by position. The window start time
   alone therefore identifies the source. The summary baseline has no clock
   information, so the gate compares against a baseline that cannot see this
   shortcut.
4. **Days are personal.** Nearest-neighbor retrieval finds another day from the
   same participant about 7 times more often than chance. Summary statistics
   do almost as well (0.227), so this is a property of CGM days themselves
   rather than something the model uniquely learned. That makes
   fingerprints useful for personal-pattern research and also a
   re-identification risk. Treat fingerprints as personal data.

## Next step (a future protocol, not 1.2)

Align every canonical day to local midnight in the BIG IDEAs adapter (or add
random phase-shift augmentation), declare protocol 1.3 with a new untouched
holdout, and repeat. That separates clock-alignment artifacts from genuine
source differences.
