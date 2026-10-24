"""
Paket Modul SIGAP (Smarter Protection for Sustainable JKN)
Berisi modul untuk preprocessing, pelatihan model ML, dan evaluasi risiko.
"""

from .preprocessing import load_and_preprocess_data, hash_value
from .train import train_risk_model
from .evaluate import evaluate_model

__version__ = "1.0.0"