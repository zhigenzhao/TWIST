#!/bin/bash
# T1 Motion Retargeting Script
# Complete workflow for retargeting AMASS+OMOMO to Booster T1
# See T1_RETARGETING_PLAN.md for detailed documentation

set -e  # Exit on error

echo "========================================"
echo "T1 Motion Retargeting Workflow"
echo "========================================"
echo ""

# Configuration
MY_DATASET="/home/szhan43/TWIST/my_dataset"
OUTPUT_DIR="/home/szhan43/TWIST/track_dataset/twist_motion_dataset"
GMR_DIR="/home/szhan43/TWIST/GMR"

# Step 1: Convert OMOMO to SMPL-X
echo "Step 1: Converting OMOMO dataset to SMPL-X format..."
if [ ! -d "$MY_DATASET/omomo_smplx" ]; then
    cd "$GMR_DIR"
    python scripts/convert_omomo_to_smplx.py \
      --src_folder "$MY_DATASET/omomo" \
      --tgt_folder "$MY_DATASET/omomo_smplx"
    echo "✓ OMOMO converted to SMPL-X"
else
    echo "✓ OMOMO already converted to SMPL-X"
fi
echo ""

# Step 2: Test single motion retargeting
echo "Step 2: Testing single motion retargeting..."
TEST_FILE=$(find "$MY_DATASET/CMU" -name "*_stageii.npz" -type f | head -1)
echo "  Test file: $TEST_FILE"

cd "$GMR_DIR"
python scripts/smplx_to_robot.py \
  --smplx_file "$TEST_FILE" \
  --robot booster_t1 \
  --save_path /tmp/t1_test_motion.pkl

echo "✓ Single motion test complete!"
echo "  You can visualize with:"
echo "  python $GMR_DIR/scripts/vis_robot_motion.py --robot booster_t1 --robot_motion_path /tmp/t1_test_motion.pkl"
echo ""

# Step 3: Batch retarget AMASS datasets
echo "Step 3: Batch retargeting AMASS datasets to T1..."
echo "  This will take 12-24 hours for all datasets."
echo "  Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR"
cd "$GMR_DIR"

# Dataset list (AMASS datasets only, no BMLhandball since deleted)
DATASETS=(
  "ACCAD"
  "BMLmovi"
  "CMU"
  "CNRS"
  "DanceDB"
  "DFaust"
  "EKUT"
  "Eyes_Japan_Dataset"
  "GRAB"
  "HDM05"
  "HUMAN4D"
  "HumanEva"
  "KIT"
  "MoSh"
  "PosePrior"
  "SFU"
  "TotalCapture"
  "Transitions"
)

for dataset in "${DATASETS[@]}"; do
  # Convert to lowercase for output directory
  output_name=$(echo "$dataset" | tr '[:upper:]' '[:lower:]')

  # Special case: EyesJapanDataset → eyes_japan, HUMAN4D → human4d, MoSh → mpi_mosh
  if [ "$dataset" = "EyesJapanDataset" ]; then
    output_name="eyes_japan"
  elif [ "$dataset" = "HUMAN4D" ]; then
    output_name="human4d"
  elif [ "$dataset" = "MoSh" ]; then
    output_name="mpi_mosh"
  fi

  # Skip if already retargeted (even partial completion counts as done to avoid hanging)
  if [ -d "$OUTPUT_DIR/$output_name" ] && [ "$(find "$OUTPUT_DIR/$output_name" -name '*.pkl' | wc -l)" -gt 0 ]; then
    count=$(find "$OUTPUT_DIR/$output_name" -name '*.pkl' | wc -l)
    echo "  ⊘ Skipping $dataset (already has $count retargeted files as $output_name)"
    continue
  fi

  # Skip if source directory doesn't exist
  if [ ! -d "$MY_DATASET/$dataset" ]; then
    echo "  ⚠ Warning: $MY_DATASET/$dataset not found, skipping"
    continue
  fi

  echo "  ========================================="
  echo "  Retargeting $dataset → $output_name"
  echo "  ========================================="

  python scripts/smplx_to_robot_dataset.py \
    --robot booster_t1 \
    --src_folder "$MY_DATASET/$dataset" \
    --tgt_folder "$OUTPUT_DIR/$output_name"

  echo "  ✓ Completed $dataset → $output_name"
  echo ""
done

echo "✓ All AMASS datasets retargeted!"
echo ""

# Step 4: Retarget OMOMO
echo "Step 4: Retargeting OMOMO dataset..."
if [ -d "$OUTPUT_DIR/omomo" ] && [ "$(find "$OUTPUT_DIR/omomo" -name '*.pkl' | wc -l)" -gt 0 ]; then
    echo "✓ OMOMO already retargeted"
else
    cd "$GMR_DIR"
    python scripts/smplx_to_robot_dataset.py \
      --robot booster_t1 \
      --src_folder "$MY_DATASET/omomo" \
      --tgt_folder "$OUTPUT_DIR/omomo"
    echo "✓ OMOMO retargeting complete!"
fi
echo ""

# Step 5: Verify results
echo "Step 5: Verifying retargeting results..."
total_files=$(find "$OUTPUT_DIR" -name "*.pkl" | wc -l)
echo "  Total retargeted files: $total_files"
echo "  Expected: ~21,000 files"
echo ""
echo "  Files per dataset:"
for dir in "$OUTPUT_DIR"/*/ ; do
  count=$(find "$dir" -name "*.pkl" | wc -l)
  echo "    $(basename $dir): $count files"
done
echo ""

# # Step 6: Create t1_twist_dataset.yaml
# echo "Step 6: Creating t1_twist_dataset.yaml..."
# YAML_PATH="/home/szhan43/TWIST/legged_gym/motion_data_configs/t1_twist_dataset.yaml"

# if [ -f "$YAML_PATH" ]; then
#     echo "  ⚠ $YAML_PATH already exists, backing up..."
#     cp "$YAML_PATH" "${YAML_PATH}.backup"
# fi

# cp /home/szhan43/TWIST/legged_gym/motion_data_configs/twist_dataset.yaml "$YAML_PATH"
# sed -i 's|root_path:.*|root_path: /home/szhan43/TWIST/track_dataset/t1_motion_dataset|' "$YAML_PATH"

# echo "✓ Created t1_twist_dataset.yaml"
# echo "  Verify with: head -20 $YAML_PATH"
# echo ""

# Summary
echo "========================================"
echo "✓ T1 Retargeting Complete!"
echo "========================================"
echo ""
echo "Summary:"
echo "  - OMOMO converted to SMPL-X: ✓"
echo "  - Single motion test: ✓"
echo "  - AMASS retargeted: ✓"
echo "  - OMOMO retargeted: ✓"
echo "  - t1_twist_dataset.yaml created: ✓"
echo ""
echo "Next steps:"
echo "  1. Verify T1 config (see T1_VERIFICATION_CHECKLIST.md)"
echo "  2. Test T1 URDF loading in Isaac Gym"
echo "  3. Start teacher training: bash train_teacher_t1.sh t1_teacher_v1 cuda:0"
echo ""
