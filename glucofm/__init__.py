"""Small research components for encoding CGM time series."""

from .canonical import CGMDay, CGMReading, SourceProvenance, build_24h_windows
from .corpus import (
    CanonicalCGMDataset,
    MultiSourceCGMDataset,
    SourceBalancedSampler,
    create_participant_split,
    create_prospective_holdout_split,
)
from .data import CGMSeries, CGMWindowDataset, load_cgm_csv
from .inference import ResearchEncoder, cosine_similarity, search_manifests
from .model import GlucoFM, GlucoFMConfig

__all__ = [
    "CGMSeries",
    "CGMDay",
    "CGMReading",
    "CGMWindowDataset",
    "CanonicalCGMDataset",
    "MultiSourceCGMDataset",
    "SourceBalancedSampler",
    "GlucoFM",
    "GlucoFMConfig",
    "ResearchEncoder",
    "SourceProvenance",
    "build_24h_windows",
    "create_participant_split",
    "create_prospective_holdout_split",
    "cosine_similarity",
    "load_cgm_csv",
    "search_manifests",
]

__version__ = "0.1.0"
