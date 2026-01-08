#!/bin/bash
# Full training pipeline script
# Trains DecisionNN, then CurveLSTM, then evaluates everything

set -e

echo "=========================================="
echo "FULL AI DJ TRAINING PIPELINE"
echo "=========================================="

cd "$(dirname "$0")/.."
source venv/bin/activate

# Stage 1: Train DecisionNN
echo ""
echo "=========================================="
echo "STAGE 1: Training DecisionNN"
echo "=========================================="
python scripts/train_model.py --stage 1 --epochs 100 --batch-size 32

# Stage 2: Train CurveLSTM
echo ""
echo "=========================================="
echo "STAGE 2: Training CurveLSTM"
echo "=========================================="
python scripts/train_model.py --stage 2 --epochs 50 --batch-size 16

# Evaluation: DecisionNN
echo ""
echo "=========================================="
echo "EVALUATION: DecisionNN"
echo "=========================================="
python scripts/evaluate_model.py --model models/decision_nn.pt --data data/training_splits/test.json

# Evaluation: End-to-end
echo ""
echo "=========================================="
echo "EVALUATION: End-to-End Combined Model"
echo "=========================================="
python scripts/end_to_end_test.py

echo ""
echo "=========================================="
echo "TRAINING PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Models saved to: models/"
echo "  - decision_nn.pt"
echo "  - curve_lstm.pt"
echo ""
echo "Evaluation results:"
echo "  - models/evaluation_results.json"
echo "  - models/end_to_end_results.json"

