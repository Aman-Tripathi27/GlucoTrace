# Data sources and canonical format

This project does not redistribute CGM datasets. Researchers obtain each source
under its own terms, run a source-specific adapter locally, and retain the
generated provenance manifest with every experiment.

## First supported source: BIG IDEAs

The [BIG IDEAs Lab Glycemic Variability and Wearable Device Data](https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/)
dataset is hosted by PhysioNet. Version 1.1.3 contains 16 participant folders.
Each folder has a `Dexcom_<participant>.csv` file alongside much larger
high-frequency wearable files. This project reads only the Dexcom files.

- Dataset version: 1.1.3
- Device: Dexcom G6
- Native CGM interval: approximately 5 minutes
- Glucose unit: mg/dL
- File license: Open Data Commons Attribution License 1.0
- Required source columns: `Event Type`,
  `Timestamp (YYYY-MM-DDThh:mm:ss)`, and `Glucose Value (mg/dL)`
- Physical measurements: rows whose event type is `EGV`

The source dates are shifted for de-identification. They still retain temporal
spacing and time-of-day, but must not be interpreted as actual collection dates.

## Obtaining the source files

Review and accept the dataset's license on PhysioNet. Download the 16 small
`Dexcom_*.csv` files into the official participant-folder layout:

```text
big-ideas-1.1.3/
├── 001/Dexcom_001.csv
├── 002/Dexcom_002.csv
├── ...
└── 016/Dexcom_016.csv
```

There is no need to download ACC, BVP, EDA, HR, IBI, or temperature files for
the CGM-only model.

## Preparing canonical days

After installing the package, run:

```bash
glucofm-prepare-big-ideas big-ideas-1.1.3 prepared/big-ideas
```

The adapter:

1. Reads only physical `EGV` rows.
2. Splits the trace when consecutive readings are more than 60 minutes apart.
3. Builds complete 24-hour windows with 288 five-minute positions.
4. Rejects windows with less than 80% observed coverage by default.
5. Zero-fills missing tensor positions without marking them observed.
6. Calculates minutes since the last physical measurement and circular
   time-of-day features.
7. Writes one canonical CSV per day and a `manifest.json` containing dataset,
   participant, device, unit, version, license, observed fraction, and SHA-256.

Canonical day CSVs contain:

```csv
timestamp,glucose,observed_mask,gap_age_minutes
2020-02-13T17:23:32,104.0,1,0.0
2020-02-13T17:28:32,,0,5.0
```

An empty glucose cell is a tensor-construction placeholder. The observation
mask—not the filled value—defines whether a measurement exists.

## Tests versus training data

Tests generate a few rows with the exact BIG IDEAs column names and folder
layout. These fixtures test parsing and missingness rules; they are not used for
training and do not attempt to simulate human physiology. Trained checkpoints
and evaluations will use only declared research datasets.

## Second supported source: Colas 2019

The authoritative source is [Supporting File S1](https://doi.org/10.1371/journal.pone.0225817.s001)
from Colás et al., “Detrended Fluctuation Analysis in the prediction of type 2
diabetes mellitus in patients at risk.” The [PLOS article](https://doi.org/10.1371/journal.pone.0225817)
states that the data are provided under the Creative Commons Attribution 4.0
International license.

- Source identifier: `10.1371/journal.pone.0225817.s001`
- Participants in the source archive: 208
- Device reported by the paper: Medtronic MiniMed iPro CGMS
- Native interval: nominally 5 minutes
- Glucose unit: mg/dL
- Source layout: `S1/case  1.csv` through `S1/case  208.csv`
- Native columns: row index, `hora` (clock time), and `glucemia`
- Missing marker: `NA`
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)

Download the ZIP from the paper’s Supporting Information section, then run:

```bash
glucofm-prepare-colas pone.0225817.s001.zip prepared/colas
```

The native case files have clock time but no calendar date. The adapter uses the
first clock value for circadian phase and reconstructs subsequent timestamps
from the consecutive source row index at exactly five-minute spacing. It uses
`2000-01-01` as a relative placeholder date and must not be interpreted as the
collection date. Source clocks can shift during missing runs, so every clock is
checked against the nominal row grid with a maximum tolerance of one sampling
interval.

The adapter converts `NA` to an unobserved mask position without interpolation.
It deliberately does not read `clinical_data.txt`: demographic, laboratory,
follow-up, and outcome fields are unnecessary for CGM representation learning.

Reference audit for the official ZIP with SHA-256
`4a40842cfc44473e54f05ece12b970f4c85e41e7ec11df0b1b943ac84eb2611b`:

- 208 source participants: 191 with 576 slots and 17 with 288 slots
- 114,912 total slots and 659 explicit `NA` positions
- 391 accepted non-overlapping days from 206 participants
- 99.61% observed coverage in accepted days
- Cases 148 and 150 excluded because gaps longer than 60 minutes split their
  one-day traces; six two-day cases contribute one accepted day for the same
  declared rule

These are mechanical ingestion statistics, not performance or clinical
findings.

## Planned source roles

- BIG IDEAs, Colas, selected ShanghaiT2DM sessions, and an eligible Stanford
  subset: candidate self-supervised pretraining sources.
- CGMacros: initially reserved for paired Dexcom/Libre representation tests.
- Google/Fitbit Wear-CGM: not used because it is non-public.

All final splits must be participant-disjoint and recorded in the manifest.

## Participant-disjoint splits

After preparing a canonical corpus, create its split file with:

```bash
glucofm-split-corpus prepared/big-ideas/manifest.json
```

The default creates a deterministic SHA-256 ordering from the seed and each
`(dataset, participant_id)` group. It records train, validation, and test
membership in `splits.json`, together with the seed, requested fractions,
record counts, and source manifest checksum. The loader refuses to combine a
split file with a changed manifest.

```python
from glucofm import CanonicalCGMDataset

validation_days = CanonicalCGMDataset(
    "prepared/big-ideas/manifest.json",
    split_path="prepared/big-ideas/splits.json",
    split="validation",
)
```

Participant identifiers are scoped to their declared dataset. If two sources
could contain the same human under unrelated identifiers, that relationship
must be resolved through source governance; this software cannot infer it.
