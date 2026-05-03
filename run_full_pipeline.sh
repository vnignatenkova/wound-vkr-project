#!/bin/bash
set -e

python3 -m src.segmentation.segmentation
python3 -m src.features.extract_wound_features
python3 -m src.features.build_wound_dynamics
python3 -m src.modeling.analyze_feature_significance
python3 -m src.modeling.compare_feature_sets
python3 -m src.modeling.prune_feature_sets
python3 -m src.modeling.train_compact_healing_model
python3 -m src.modeling.predict_compact_healing_model
python3 -m src.modeling.build_compact_healing_report
