cat > run_full_pipeline.sh <<'SH'
#!/bin/bash
set -e

python3 segmentation.py
python3 extract_wound_features.py
python3 build_wound_dynamics.py
python3 analyze_feature_significance.py
python3 compare_feature_sets.py
python3 prune_feature_sets.py
python3 train_compact_healing_model.py
python3 predict_compact_healing_model.py
python3 build_compact_healing_report.py
SH

chmod +x run_full_pipeline.sh
./run_full_pipeline.sh