#!/bin/bash
# Grid Search Script for EM Parameters
# Usage:
#   SEARCH_TAU_AGENTFAIL=1 SEARCH_TAU_ENVFAIL=1 ./test_param.sh   # Search tau params only
#   SEARCH_W_GUI=1 SEARCH_W_CODE=1 ./test_param.sh                # Search weight params only
#   SEARCH_ALL=1 ./test_param.sh                                   # Search all params (slow)
#   ./test_param.sh                                                # Run with defaults only

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Output file
OUTPUT_FILE="${OUTPUT_FILE:-grid_search_results.jsonl}"
# Clear previous results if exists
> "$OUTPUT_FILE"

# Default values (used when parameter is not being searched)
DEFAULT_TAU_AGENTFAIL=0.75
DEFAULT_TAU_ENVFAIL=0.75
DEFAULT_W_GUI=1.0
DEFAULT_W_CODE=1.2
DEFAULT_W_NORESP=0.3
DEFAULT_AGENT_WEIGHT=0.9

# Search ranges for each parameter
TAU_AGENTFAIL_RANGE=(0.5 0.6 0.7 0.75 0.8 0.9)
TAU_ENVFAIL_RANGE=(0.5 0.6 0.7 0.75 0.8 0.9)
W_GUI_RANGE=(0.5 0.8 1.0 1.2 1.5)
W_CODE_RANGE=(0.8 1.0 1.2 1.5 2.0)
W_NORESP_RANGE=(0.1 0.3 0.5 0.8)
AGENT_WEIGHT_RANGE=(0.5 0.7 0.9 1.0)

# Determine which parameters to search based on environment variables
if [[ -n "$SEARCH_ALL" ]]; then
    SEARCH_TAU_AGENTFAIL=1
    SEARCH_TAU_ENVFAIL=1
    SEARCH_W_GUI=1
    SEARCH_W_CODE=1
    SEARCH_W_NORESP=1
    SEARCH_AGENT_WEIGHT=1
fi

# Build arrays for each parameter (either search range or single default)
if [[ -n "$SEARCH_TAU_AGENTFAIL" ]]; then
    tau_agentfail_values=("${TAU_AGENTFAIL_RANGE[@]}")
else
    tau_agentfail_values=("$DEFAULT_TAU_AGENTFAIL")
fi

if [[ -n "$SEARCH_TAU_ENVFAIL" ]]; then
    tau_envfail_values=("${TAU_ENVFAIL_RANGE[@]}")
else
    tau_envfail_values=("$DEFAULT_TAU_ENVFAIL")
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

# Calculate total combinations
total_combos=$((${#tau_agentfail_values[@]} * ${#tau_envfail_values[@]} * ${#w_gui_values[@]} * ${#w_code_values[@]} * ${#w_noresp_values[@]} * ${#agent_weight_values[@]}))
echo "Total combinations to test: $total_combos"
echo "Results will be saved to: $OUTPUT_FILE"
echo ""

# Function to parse metrics from output
parse_metrics() {
    local output="$1"
    
    # Parse all 4 accuracy values (format: "Accuracy: 0.xxx")
    # First 2 from main() (test), next 2 from run_prediction() (val)
    local acc_values=$(echo "$output" | grep -oP 'Accuracy: \K[0-9.]+')
    local test_acc_original=$(echo "$acc_values" | sed -n '1p')
    local test_acc_correct=$(echo "$acc_values" | sed -n '2p')
    local val_acc_original=$(echo "$acc_values" | sed -n '3p')
    local val_acc_correct=$(echo "$acc_values" | sed -n '4p')
    
    # Parse Subset A: misflipped count (format: "[Subset A] 原判断正确: X cases, 被误翻 Y")
    # First occurrence is test, second is val
    local subset_a_lines=$(echo "$output" | grep -o '\[Subset A\].*被误翻 [0-9]*')
    local test_subset_a_line=$(echo "$subset_a_lines" | sed -n '1p')
    local val_subset_a_line=$(echo "$subset_a_lines" | sed -n '2p')
    
    local test_subset_a_total=$(echo "$test_subset_a_line" | grep -oP '原判断正确: \K[0-9]+')
    local test_misflip_count=$(echo "$test_subset_a_line" | grep -oP '被误翻 \K[0-9]+')
    local val_subset_a_total=$(echo "$val_subset_a_line" | grep -oP '原判断正确: \K[0-9]+')
    local val_misflip_count=$(echo "$val_subset_a_line" | grep -oP '被误翻 \K[0-9]+')
    
    # Parse Subset B: corrected count (format: "[Subset B] 原判断错误: X cases, 被成功纠正 Y")
    # First occurrence is test, second is val
    local subset_b_lines=$(echo "$output" | grep -o '\[Subset B\].*被成功纠正 [0-9]*')
    local test_subset_b_line=$(echo "$subset_b_lines" | sed -n '1p')
    local val_subset_b_line=$(echo "$subset_b_lines" | sed -n '2p')
    
    local test_subset_b_total=$(echo "$test_subset_b_line" | grep -oP '原判断错误: \K[0-9]+')
    local test_correction_count=$(echo "$test_subset_b_line" | grep -oP '被成功纠正 \K[0-9]+')
    local val_subset_b_total=$(echo "$val_subset_b_line" | grep -oP '原判断错误: \K[0-9]+')
    local val_correction_count=$(echo "$val_subset_b_line" | grep -oP '被成功纠正 \K[0-9]+')
    
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
    
    echo "$val_acc_original $val_acc_correct $test_acc_original $test_acc_correct $val_subset_a_total $val_misflip_count $val_subset_b_total $val_correction_count $test_subset_a_total $test_misflip_count $test_subset_b_total $test_correction_count"
}

# Counter for progress
counter=0

# Grid search loop
for tau_agentfail in "${tau_agentfail_values[@]}"; do
    for tau_envfail in "${tau_envfail_values[@]}"; do
        for w_gui in "${w_gui_values[@]}"; do
            for w_code in "${w_code_values[@]}"; do
                for w_noresp in "${w_noresp_values[@]}"; do
                    for agent_weight in "${agent_weight_values[@]}"; do
                        counter=$((counter + 1))
                        echo "[$counter/$total_combos] Running: tau_agentfail=$tau_agentfail tau_envfail=$tau_envfail w_gui=$w_gui w_code=$w_code w_noresp=$w_noresp agent_weight=$agent_weight"
                        
                        # Run the Python script and capture output
                        output=$(python run_rootcase.py \
                            --tau_agentfail "$tau_agentfail" \
                            --tau_envfail "$tau_envfail" \
                            --w_gui "$w_gui" \
                            --w_code "$w_code" \
                            --w_noresp "$w_noresp" \
                            --agent_weight "$agent_weight" \
                            --out_dir "em_outputs_grid_search_tmp" \
                            2>&1) || true
                        
                        # Parse metrics from output
                        read val_acc_original val_acc_correct test_acc_original test_acc_correct val_subset_a_total val_misflip_count val_subset_b_total val_correction_count test_subset_a_total test_misflip_count test_subset_b_total test_correction_count <<< $(parse_metrics "$output")
                        
                        # Calculate rates for validation set (use awk to ensure proper decimal format)
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
                        
                        # Write JSONL entry with both val and test metrics
                        echo "{\"tau_agentfail\": $tau_agentfail, \"tau_envfail\": $tau_envfail, \"w_gui\": $w_gui, \"w_code\": $w_code, \"w_noresp\": $w_noresp, \"agent_weight\": $agent_weight, \"val_acc_original\": $val_acc_original, \"val_acc_correct\": $val_acc_correct, \"val_subset_a_total\": $val_subset_a_total, \"val_misflip_count\": $val_misflip_count, \"val_misflip_rate\": $val_misflip_rate, \"val_subset_b_total\": $val_subset_b_total, \"val_correction_count\": $val_correction_count, \"val_correction_rate\": $val_correction_rate, \"test_acc_original\": $test_acc_original, \"test_acc_correct\": $test_acc_correct, \"test_subset_a_total\": $test_subset_a_total, \"test_misflip_count\": $test_misflip_count, \"test_misflip_rate\": $test_misflip_rate, \"test_subset_b_total\": $test_subset_b_total, \"test_correction_count\": $test_correction_count, \"test_correction_rate\": $test_correction_rate}" >> "$OUTPUT_FILE"
                        
                        echo "  -> test: acc_orig=$test_acc_original, acc_corr=$test_acc_correct, misflip=$test_misflip_count/$test_subset_a_total, corrected=$test_correction_count/$test_subset_b_total"
                        echo "  -> val: acc_orig=$val_acc_original, acc_corr=$val_acc_correct, misflip=$val_misflip_count/$val_subset_a_total, corrected=$val_correction_count/$val_subset_b_total"
                        echo ""
                    done
                done
            done
        done
    done
done

echo "=========================================="
echo "Grid search complete!"
echo "Results saved to: $OUTPUT_FILE"
echo "Total experiments: $counter"

# Show best result by val_acc_correct (val = run_prediction on full dataset)
echo ""
echo "Top 5 results by val_acc_correct:"
python3 -c "
import json
results = []
with open('$OUTPUT_FILE', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            results.append(json.loads(line))
# Sort by val_acc_correct descending
results.sort(key=lambda x: x.get('val_acc_correct', 0), reverse=True)
for d in results[:5]:
    print(f'  test_acc={d[\"test_acc_correct\"]:.4f} val_acc={d[\"val_acc_correct\"]:.4f} | tau_af={d[\"tau_agentfail\"]} tau_ef={d[\"tau_envfail\"]} w_gui={d[\"w_gui\"]} w_code={d[\"w_code\"]} w_noresp={d[\"w_noresp\"]} agent_w={d[\"agent_weight\"]}')
"

