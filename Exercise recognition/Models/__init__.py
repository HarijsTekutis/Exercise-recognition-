"""Models package for exercise recognition."""

from .CNN_ResBiLSTM import CNNLSTM, train_cnnlstm, evaluate_cnnlstm
from .Multi_head_CNN_ResBiLSTM import MULTI_HEAD_CNN_LSTM, train_multi_head_cnn_lstm, evaluate_multi_head_cnn_lstm
from .RandomForest import RandomForest, RandomForestModel, evaluate_random_forest, train_random_forest
from .Rocket import (Rocket,RocketModel,evaluate_rocket,train_rocket)
from .CNN_ResBiGru import CNNResBiGRU, train_cnn_resbigru, evaluate_cnn_resbigru
from .XGBoost import XGBoost, XGBoostModel, evaluate_xgboost, train_xgboost

__all__ = [
    "CNNLSTM",
    "MULTI_HEAD_CNN_LSTM",
    "RandomForest",
    "RandomForestModel",
    "Rocket",
    "RocketModel",
    "XGBoost",
    "XGBoostModel",
    "train_cnnlstm",
    "evaluate_cnnlstm",
    "train_multi_head_cnn_lstm",
    "evaluate_multi_head_cnn_lstm",
    "train_random_forest",
    "evaluate_random_forest",
    "train_rocket",
    "evaluate_rocket",
    "train_xgboost",
    "evaluate_xgboost",
]
