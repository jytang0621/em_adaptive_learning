#!/bin/bash
# Grid Search Script for EM Parameters (Parallel Version)
# Usage:
#   SEARCH_MARGIN_AGENTFAIL=1 SEARCH_TAU_ENVFAIL_HIGH=1 ./test_param.sh   # Search threshold params only
#   SEARCH_TAU_AGENTFAIL_HIGH=1 SEARCH_TAU_SUPPORT=1 ./test_param.sh      # Search additional threshold params
#   SEARCH_TAU_AGENTFAIL_FLOOR=1 ./test_param.sh                          # Search agent fail floor threshold
#   SEARCH_TAU_SUPPORT_POS=1 SEARCH_TAU_SUPPORT_NEG=1 ./test_param.sh     # Search asymmetric support thresholds
#   SEARCH_MIN_NEG_CHANNELS=1 ./test_param.sh                             # Search multi-channel consistency
#   SEARCH_W_GUI=1 SEARCH_W_CODE=1 ./test_param.sh                        # Search weight params only
#   SEARCH_TEMP=1 ./test_param.sh                                         # Search temperature only (1.0/1.2/1.5)
#   SEARCH_ALL=1 ./test_param.sh                                          # Search all params (slow)
#   NUM_WORKERS=4 ./test_param.sh                                         # Control parallelism (default: 8)
#   ./test_param.sh                                                       # Run with defaults only

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Parallel workers (default: 8)
NUM_WORKERS="${NUM_WORKERS:-8}"

# Output file
OUTPUT_FILE="${OUTPUT_FILE:-grid_search_results.jsonl}"
OUTPUT_FILE_ABS="$SCRIPT_DIR/$OUTPUT_FILE"
LOCK_FILE="$OUTPUT_FILE_ABS.lock"
COUNTER_FILE="$OUTPUT_FILE_ABS.counter"

# Clear previous results if exists
> "$OUTPUT_FILE_ABS"
echo "0" > "$COUNTER_FILE"

# Default values (used when parameter is not being searched)
DEFAULT_MARGIN_AGENTFAIL=0.6
DEFAULT_TAU_ENVFAIL_HIGH=0.72
DEFAULT_TAU_AGENTFAIL_HIGH=0.95
DEFAULT_TAU_SUPPORT=0.0
DEFAULT_W_GUI=1.2
DEFAULT_W_CODE=1.2
DEFAULT_W_NORESP=0.15
DEFAULT_AGENT_WEIGHT=0.45
DEFAULT_TEMP=1.0
DEFAULT_TAU_SUPPORT_POS=0.0
DEFAULT_TAU_SUPPORT_NEG=0.0
DEFAULT_MIN_NEG_CHANNELS=0
DEFAULT_TAU_AGENTFAIL_FLOOR=0.65

MARGIN_AGENTFAIL_RANGE=(0.25 0.6)
TAU_AGENTFAIL_FLOOR_RANGE=(0.40, 0.45, 0.50, 0.55)
TAU_AGENTFAIL_HIGH_RANGE=(1.00)

# 0->1 证据主门槛：你现在的改动明显降低误翻，所以把范围集中在 0.15~0.35
TAU_SUPPORT_POS_RANGE=(0.08)

# 1->0 更严格：建议从"比 pos 更严"出发
TAU_SUPPORT_NEG_RANGE=(0.20)

MIN_NEG_CHANNELS_RANGE=(1)   # 0 太松，容易把正确的 PASS 翻掉
TAU_ENVFAIL_HIGH_RANGE=(0.72)
W_GUI_RANGE=(0.8)
W_CODE_RANGE=(1.0)
W_NORESP_RANGE=(0.05)
AGENT_WEIGHT_RANGE=(0.20)
TEMP_RANGE=(1)


# Determine which parameters to search based on environment variables
if [[ -n "$SEARCH_ALL" ]]; then
    SEARCH_MARGIN_AGENTFAIL=1
    SEARCH_TAU_ENVFAIL_HIGH=1
    SEARCH_TAU_AGENTFAIL_HIGH=1
    # Note: SEARCH_TAU_SUPPORT removed - replaced by asymmetric TAU_SUPPORT_POS/NEG
    SEARCH_W_GUI=1
    SEARCH_W_CODE=1
    SEARCH_W_NORESP=1
    SEARCH_AGENT_WEIGHT=1
    SEARCH_TEMP=1
    SEARCH_TAU_SUPPORT_POS=1
    SEARCH_TAU_SUPPORT_NEG=1
    SEARCH_MIN_NEG_CHANNELS=1
    SEARCH_TAU_AGENTFAIL_FLOOR=1
fi

# Build arrays for each parameter (either search range or single default)
if [[ -n "$SEARCH_MARGIN_AGENTFAIL" ]]; then
    margin_agentfail_values=("${MARGIN_AGENTFAIL_RANGE[@]}")
else
    margin_agentfail_values=("$DEFAULT_MARGIN_AGENTFAIL")
fi

if [[ -n "$SEARCH_TAU_ENVFAIL_HIGH" ]]; then
    tau_envfail_high_values=("${TAU_ENVFAIL_HIGH_RANGE[@]}")
else
    tau_envfail_high_values=("$DEFAULT_TAU_ENVFAIL_HIGH")
fi

if [[ -n "$SEARCH_TAU_AGENTFAIL_HIGH" ]]; then
    tau_agentfail_high_values=("${TAU_AGENTFAIL_HIGH_RANGE[@]}")
else
    tau_agentfail_high_values=("$DEFAULT_TAU_AGENTFAIL_HIGH")
fi

if [[ -n "$SEARCH_TAU_SUPPORT" ]]; then
    tau_support_values=("${TAU_SUPPORT_RANGE[@]}")
else
    tau_support_values=("$DEFAULT_TAU_SUPPORT")
fi

if [[ -n "$SEARCH_W_GUI" ]]; then
    w_gui_values=("${W_GUI_RANGE[@]}")
else
    w_gui_values=("$DEFAULT_W_GUI")
fi

if [[ -n "$SEARCH_W_CODE" ]]; then
    w_code_values=("${W_CODE_RANGE[@]}")
else
    w_code_values=("$DEFAULT_W_CODE")
fi

if [[ -n "$SEARCH_W_NORESP" ]]; then
    w_noresp_values=("${W_NORESP_RANGE[@]}")
else
    w_noresp_values=("$DEFAULT_W_NORESP")
fi

if [[ -n "$SEARCH_AGENT_WEIGHT" ]]; then
    agent_weight_values=("${AGENT_WEIGHT_RANGE[@]}")
else
    agent_weight_values=("$DEFAULT_AGENT_WEIGHT")
fi

if [[ -n "$SEARCH_TEMP" ]]; then
    temp_values=("${TEMP_RANGE[@]}")
else
    temp_values=("$DEFAULT_TEMP")
fi

if [[ -n "$SEARCH_TAU_SUPPORT_POS" ]]; then
    tau_support_pos_values=("${TAU_SUPPORT_POS_RANGE[@]}")
else
    tau_support_pos_values=("$DEFAULT_TAU_SUPPORT_POS")
fi

if [[ -n "$SEARCH_TAU_SUPPORT_NEG" ]]; then
    tau_support_neg_values=("${TAU_SUPPORT_NEG_RANGE[@]}")
else
    tau_support_neg_values=("$DEFAULT_TAU_SUPPORT_NEG")
fi

if [[ -n "$SEARCH_MIN_NEG_CHANNELS" ]]; then
    min_neg_channels_values=("${MIN_NEG_CHANNELS_RANGE[@]}")
else
    min_neg_channels_values=("$DEFAULT_MIN_NEG_CHANNELS")
fi

if [[ -n "$SEARCH_TAU_AGENTFAIL_FLOOR" ]]; then
    tau_agentfail_floor_values=("${TAU_AGENTFAIL_FLOOR_RANGE[@]}")
else
    tau_agentfail_floor_values=("$DEFAULT_TAU_AGENTFAIL_FLOOR")
fi

# Calculate total combinations
total_combos=$((${#margin_agentfail_values[@]} * ${#tau_envfail_high_values[@]} * ${#tau_agentfail_high_values[@]} * ${#tau_support_values[@]} * ${#w_gui_values[@]} * ${#w_code_values[@]} * ${#w_noresp_values[@]} * ${#agent_weight_values[@]} * ${#temp_values[@]} * ${#tau_support_pos_values[@]} * ${#tau_support_neg_values[@]} * ${#min_neg_channels_values[@]} * ${#tau_agentfail_floor_values[@]}))
echo "Total combinations to test: $total_combos"
echo "Parallel workers: $NUM_WORKERS"
echo "Results will be saved to: $OUTPUT_FILE"
echo ""

# Function to generate all parameter combinations
generate_params() {
    for margin_agentfail in "${margin_agentfail_values[@]}"; do
        for tau_envfail_high in "${tau_envfail_high_values[@]}"; do
            for tau_agentfail_high in "${tau_agentfail_high_values[@]}"; do
                for tau_support in "${tau_support_values[@]}"; do
                    for w_gui in "${w_gui_values[@]}"; do
                        for w_code in "${w_code_values[@]}"; do
                            for w_noresp in "${w_noresp_values[@]}"; do
                                for agent_weight in "${agent_weight_values[@]}"; do
                                    for temp in "${temp_values[@]}"; do
                                        for tau_support_pos in "${tau_support_pos_values[@]}"; do
                                            for tau_support_neg in "${tau_support_neg_values[@]}"; do
                                                for min_neg_channels in "${min_neg_channels_values[@]}"; do
                                                    for tau_agentfail_floor in "${tau_agentfail_floor_values[@]}"; do
                                                        # Output parameters as pipe-separated string
                                                        echo "$margin_agentfail|$tau_envfail_high|$tau_agentfail_high|$tau_agentfail_floor|$tau_support|$w_gui|$w_code|$w_noresp|$agent_weight|$temp|$tau_support_pos|$tau_support_neg|$min_neg_channels"
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
}

# Function to run a single experiment (called in parallel)
run_single_experiment() {
    local params="$1"
    local output_file="$2"
    local lock_file="$3"
    local counter_file="$4"
    local total="$5"
    local script_dir="$6"
    
    # Parse pipe-separated parameters
    IFS='|' read -r margin_agentfail tau_envfail_high tau_agentfail_high tau_agentfail_floor tau_support w_gui w_code w_noresp agent_weight temp tau_support_pos tau_support_neg min_neg_channels <<< "$params"
    
    # Increment counter atomically
    local current
    {
        flock -x 200
        current=$(cat "$counter_file")
        current=$((current + 1))
        echo "$current" > "$counter_file"
    } 200>"$counter_file.lock"
    
    echo "[$current/$total] Running: margin_agentfail=$margin_agentfail tau_envfail_high=$tau_envfail_high tau_agentfail_high=$tau_agentfail_high tau_agentfail_floor=$tau_agentfail_floor tau_support=$tau_support w_gui=$w_gui w_code=$w_code w_noresp=$w_noresp agent_weight=$agent_weight temp=$temp tau_support_pos=$tau_support_pos tau_support_neg=$tau_support_neg min_neg_channels=$min_neg_channels"
    
    # Run the Python script and capture output
    cd "$script_dir"
    output=$(python run_rootcase.py \
        --margin_agentfail "$margin_agentfail" \
        --tau_envfail_high "$tau_envfail_high" \
        --tau_agentfail_high "$tau_agentfail_high" \
        --tau_agentfail_floor "$tau_agentfail_floor" \
        --tau_support "$tau_support" \
        --w_gui "$w_gui" \
        --w_code "$w_code" \
        --w_noresp "$w_noresp" \
        --agent_weight "$agent_weight" \
        --temp "$temp" \
        --tau_support_pos "$tau_support_pos" \
        --tau_support_neg "$tau_support_neg" \
        --min_neg_channels "$min_neg_channels" \
        --out_dir "em_outputs_grid_search_tmp_$$" \
        2>&1) || true
    
    # Clean up temp output dir
    rm -rf "em_outputs_grid_search_tmp_$$" 2>/dev/null || true
    
    # Parse metrics from output
    # Parse all 4 accuracy values (format: "Accuracy: 0.xxx")
    acc_values=$(echo "$output" | grep -oP 'Accuracy: \K[0-9.]+' || echo "")
    test_acc_original=$(echo "$acc_values" | sed -n '1p')
    test_acc_correct=$(echo "$acc_values" | sed -n '2p')
    val_acc_original=$(echo "$acc_values" | sed -n '3p')
    val_acc_correct=$(echo "$acc_values" | sed -n '4p')
    
    # Parse Subset A: misflipped count
    subset_a_lines=$(echo "$output" | grep -o '\[Subset A\].*被误翻 [0-9]*' || echo "")
    test_subset_a_line=$(echo "$subset_a_lines" | sed -n '1p')
    val_subset_a_line=$(echo "$subset_a_lines" | sed -n '2p')
    
    test_subset_a_total=$(echo "$test_subset_a_line" | grep -oP '原判断正确: \K[0-9]+' || echo "0")
    test_misflip_count=$(echo "$test_subset_a_line" | grep -oP '被误翻 \K[0-9]+' || echo "0")
    val_subset_a_total=$(echo "$val_subset_a_line" | grep -oP '原判断正确: \K[0-9]+' || echo "0")
    val_misflip_count=$(echo "$val_subset_a_line" | grep -oP '被误翻 \K[0-9]+' || echo "0")
    
    # Parse Subset B: corrected count
    subset_b_lines=$(echo "$output" | grep -o '\[Subset B\].*被成功纠正 [0-9]*' || echo "")
    test_subset_b_line=$(echo "$subset_b_lines" | sed -n '1p')
    val_subset_b_line=$(echo "$subset_b_lines" | sed -n '2p')
    
    test_subset_b_total=$(echo "$test_subset_b_line" | grep -oP '原判断错误: \K[0-9]+' || echo "0")
    test_correction_count=$(echo "$test_subset_b_line" | grep -oP '被成功纠正 \K[0-9]+' || echo "0")
    val_subset_b_total=$(echo "$val_subset_b_line" | grep -oP '原判断错误: \K[0-9]+' || echo "0")
    val_correction_count=$(echo "$val_subset_b_line" | grep -oP '被成功纠正 \K[0-9]+' || echo "0")
    
    # Set defaults if parsing failed
    val_acc_original=${val_acc_original:-0}
    val_acc_correct=${val_acc_correct:-0}
    test_acc_original=${test_acc_original:-0}
    test_acc_correct=${test_acc_correct:-0}
    val_subset_a_total=${val_subset_a_total:-0}
    val_misflip_count=${val_misflip_count:-0}
    val_subset_b_total=${val_subset_b_total:-0}
    val_correction_count=${val_correction_count:-0}
    test_subset_a_total=${test_subset_a_total:-0}
    test_misflip_count=${test_misflip_count:-0}
    test_subset_b_total=${test_subset_b_total:-0}
    test_correction_count=${test_correction_count:-0}
    
    # Calculate rates for validation set
    if [[ "$val_subset_a_total" -gt 0 ]]; then
        val_misflip_rate=$(awk "BEGIN {printf \"%.4f\", $val_misflip_count / $val_subset_a_total}")
    else
        val_misflip_rate=0
    fi
    
    if [[ "$val_subset_b_total" -gt 0 ]]; then
        val_correction_rate=$(awk "BEGIN {printf \"%.4f\", $val_correction_count / $val_subset_b_total}")
    else
        val_correction_rate=0
    fi
    
    # Calculate rates for test set
    if [[ "$test_subset_a_total" -gt 0 ]]; then
        test_misflip_rate=$(awk "BEGIN {printf \"%.4f\", $test_misflip_count / $test_subset_a_total}")
    else
        test_misflip_rate=0
    fi
    
    if [[ "$test_subset_b_total" -gt 0 ]]; then
        test_correction_rate=$(awk "BEGIN {printf \"%.4f\", $test_correction_count / $test_subset_b_total}")
    else
        test_correction_rate=0
    fi
    
    # Build JSON result
    json_result="{\"margin_agentfail\": $margin_agentfail, \"tau_envfail_high\": $tau_envfail_high, \"tau_agentfail_high\": $tau_agentfail_high, \"tau_agentfail_floor\": $tau_agentfail_floor, \"tau_support\": $tau_support, \"w_gui\": $w_gui, \"w_code\": $w_code, \"w_noresp\": $w_noresp, \"agent_weight\": $agent_weight, \"temp\": $temp, \"tau_support_pos\": $tau_support_pos, \"tau_support_neg\": $tau_support_neg, \"min_neg_channels\": $min_neg_channels, \"val_acc_original\": $val_acc_original, \"val_acc_correct\": $val_acc_correct, \"val_subset_a_total\": $val_subset_a_total, \"val_misflip_count\": $val_misflip_count, \"val_misflip_rate\": $val_misflip_rate, \"val_subset_b_total\": $val_subset_b_total, \"val_correction_count\": $val_correction_count, \"val_correction_rate\": $val_correction_rate, \"test_acc_original\": $test_acc_original, \"test_acc_correct\": $test_acc_correct, \"test_subset_a_total\": $test_subset_a_total, \"test_misflip_count\": $test_misflip_count, \"test_misflip_rate\": $test_misflip_rate, \"test_subset_b_total\": $test_subset_b_total, \"test_correction_count\": $test_correction_count, \"test_correction_rate\": $test_correction_rate}"
    
    # Write JSONL entry with flock for thread safety
    {
        flock -x 100
        echo "$json_result" >> "$output_file"
    } 100>"$lock_file"
    
    echo "  -> test: acc_orig=$test_acc_original, acc_corr=$test_acc_correct, misflip=$test_misflip_count/$test_subset_a_total, corrected=$test_correction_count/$test_subset_b_total"
    echo "  -> val: acc_orig=$val_acc_original, acc_corr=$val_acc_correct, misflip=$val_misflip_count/$val_subset_a_total, corrected=$val_correction_count/$val_subset_b_total"
    echo ""
}

# Export the function and required variables for xargs
export -f run_single_experiment
export OUTPUT_FILE_ABS LOCK_FILE COUNTER_FILE total_combos SCRIPT_DIR

# Run grid search in parallel using xargs
echo "Starting parallel grid search with $NUM_WORKERS workers..."
echo ""

generate_params | xargs -P "$NUM_WORKERS" -I {} bash -c 'run_single_experiment "$@"' _ {} "$OUTPUT_FILE_ABS" "$LOCK_FILE" "$COUNTER_FILE" "$total_combos" "$SCRIPT_DIR"

# Get final count
final_count=$(cat "$COUNTER_FILE")

# Cleanup temp files
rm -f "$COUNTER_FILE" "$COUNTER_FILE.lock" "$LOCK_FILE" 2>/dev/null || true

echo "=========================================="
echo "Grid search complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "Total experiments: $final_count"

# Show best result by val_acc_correct (val = run_prediction on full dataset)
echo ""
echo "Top 5 results by val_acc_correct:"
python3 -c "
import json
results = []
with open('$OUTPUT_FILE_ABS', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))
# Sort by val_acc_correct descending
results.sort(key=lambda x: x.get('val_acc_correct', 0), reverse=True)
for d in results[:5]:
    print(f'  test_acc={d[\"test_acc_correct\"]:.4f} val_acc={d[\"val_acc_correct\"]:.4f} | margin_af={d[\"margin_agentfail\"]} tau_ef={d[\"tau_envfail_high\"]} tau_af_h={d.get(\"tau_agentfail_high\", 0.95)} tau_af_f={d.get(\"tau_agentfail_floor\", 0.65)} tau_sup={d.get(\"tau_support\", 0.0)} tau_sup_pos={d.get(\"tau_support_pos\", 0.0)} tau_sup_neg={d.get(\"tau_support_neg\", 0.0)} min_neg_ch={d.get(\"min_neg_channels\", 0)} w_gui={d[\"w_gui\"]} w_code={d[\"w_code\"]} w_noresp={d[\"w_noresp\"]} agent_w={d[\"agent_weight\"]} temp={d.get(\"temp\", 1.0)}')
"
