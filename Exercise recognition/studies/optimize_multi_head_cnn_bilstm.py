#!/usr/bin/env python3
"""Study for MultiHead_CNN_BiLSTM - kept as a shortcut for `studies/run_model.py multi_head_cnn_bilstm`.

The search-and-evaluate logic used to be copy-pasted into each of these six scripts, and
the copies drifted apart. It now lives in studies/study_runner.py, driven by the model
description in studies/model_registry.py, so every architecture is trained, seeded and
scored identically.

Running this file performs the full protocol for MultiHead_CNN_BiLSTM: an Optuna search on
validation macro F1, then the top three configurations retrained over five independent
seeds each, reported as mean +/- standard deviation. Results land in
studies/results/multi_head_cnn_bilstm/.

Equivalent and preferred:
    python studies/run_model.py multi_head_cnn_bilstm
    python studies/run_all_studies.py --models multi_head_cnn_bilstm
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from studies.study_runner import run_model_study

MODEL_KEY = "multi_head_cnn_bilstm"

if __name__ == "__main__":
    run_model_study(MODEL_KEY)
