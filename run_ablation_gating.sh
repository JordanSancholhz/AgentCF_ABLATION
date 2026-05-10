#!/bin/bash

# ================================================================================
# 消融实验：门控机制对比（全部 4 个数据集）
# 每个数据集跑两个实验：
#   实验1: 属性引导 + 门控   → create_prompts + 门控（rrf + description 双评估）
#   实验2: 只有属性引导      → create_prompts1 + 直接更新（rrf 评估）
#
# 目录命名: memory_{dataset}_attr_gate / log_{dataset}_attr_gate
#           memory_{dataset}_attr      / log_{dataset}_attr
#
# 用法: bash run_ablation_gating.sh
# ================================================================================

set -e

# ===================== 自动检测 Python =====================
if [ -n "$CONDA_PREFIX" ]; then
    PYTHON_CMD="$CONDA_PREFIX/bin/python"
else
    if command -v python &> /dev/null; then
        PYTHON_CMD=python
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD=python3
    else
        echo "Python not found."
        exit 1
    fi
fi

echo "Using Python: $PYTHON_CMD"

DATASETS=("All_Beauty" "Cell_Phones" "Fashion")

echo "================================================================================"
echo "消融实验：门控机制对比"
echo "数据集: ${DATASETS[*]}"
echo "================================================================================"

TOTAL_START_TIME=$(date +%s)

# 记录每个数据集的耗时
declare -A DS_EXP1_TIME
declare -A DS_EXP2_TIME

for DATASET in "${DATASETS[@]}"; do

    echo ""
    echo "================================================================================"
    echo "==================== 数据集: $DATASET ===================="
    echo "================================================================================"

    export CURRENT_DATASET=$DATASET
    DS_START_TIME=$(date +%s)

    # ==================================================================
    # 实验1: 属性引导 + 门控
    # ==================================================================
    echo ""
    echo "################################################################################"
    echo "# [$DATASET] 实验1: 属性引导 + 门控"
    echo "#   ENABLE_ATTRIBUTE_GUIDANCE=True, ENABLE_MEMORY_GATING=True"
    echo "################################################################################"

    export ENABLE_ATTRIBUTE_GUIDANCE=True
    export ENABLE_MEMORY_GATING=True
    export EXP_SUFFIX="_attr_gate"
    export EXP_MEMORY_DIR="memory_${DATASET}_attr_gate"
    export EXP_LOG_DIR="log_${DATASET}_attr_gate"

    EXP1_START=$(date +%s)

    echo ""
    echo ">>> [$DATASET] 实验1 - 步骤 1/3: 训练"
    echo "--------------------------------------------------------------------------------"
    $PYTHON_CMD AgentCF_train_check.py

    echo ""
    echo ">>> [$DATASET] 实验1 - 步骤 2/3: 测试 (EVAL_MODE=rrf)"
    echo "--------------------------------------------------------------------------------"
    EVAL_MODE=rrf $PYTHON_CMD AgentCF_Test_log-.py

    echo ""
    echo ">>> [$DATASET] 实验1 - 步骤 3/3: 测试 (EVAL_MODE=description)"
    echo "--------------------------------------------------------------------------------"
    EVAL_MODE=description $PYTHON_CMD AgentCF_Test_log-.py

    EXP1_END=$(date +%s)
    DS_EXP1_TIME[$DATASET]=$((EXP1_END - EXP1_START))

    echo ""
    echo ">>> [$DATASET] 实验1 完成！耗时: ${DS_EXP1_TIME[$DATASET]} 秒"
    echo "    memory: memory_${DATASET}_attr_gate/"
    echo "    log:    log_${DATASET}_attr_gate/"

    # ==================================================================
    # 实验2: 只有属性引导（无门控）
    # ==================================================================
    echo ""
    echo "################################################################################"
    echo "# [$DATASET] 实验2: 只有属性引导（无门控）"
    echo "#   ENABLE_ATTRIBUTE_GUIDANCE=True, ENABLE_MEMORY_GATING=False"
    echo "################################################################################"

    export ENABLE_ATTRIBUTE_GUIDANCE=True
    export ENABLE_MEMORY_GATING=False
    export EXP_SUFFIX="_attr_only"
    export EXP_MEMORY_DIR="memory_${DATASET}_attr"
    export EXP_LOG_DIR="log_${DATASET}_attr"

    EXP2_START=$(date +%s)

    echo ""
    echo ">>> [$DATASET] 实验2 - 步骤 1/2: 训练"
    echo "--------------------------------------------------------------------------------"
    $PYTHON_CMD AgentCF_train_check.py

    echo ""
    echo ">>> [$DATASET] 实验2 - 步骤 2/2: 测试"
    echo "--------------------------------------------------------------------------------"
    $PYTHON_CMD AgentCF_Test_log-.py

    EXP2_END=$(date +%s)
    DS_EXP2_TIME[$DATASET]=$((EXP2_END - EXP2_START))

    echo ""
    echo ">>> [$DATASET] 实验2 完成！耗时: ${DS_EXP2_TIME[$DATASET]} 秒"
    echo "    memory: memory_${DATASET}_attr/"
    echo "    log:    log_${DATASET}_attr/"

    DS_END_TIME=$(date +%s)
    DS_ELAPSED=$((DS_END_TIME - DS_START_TIME))

    echo ""
    echo "================================================================================"
    echo "[$DATASET] 全部完成！总耗时: ${DS_ELAPSED} 秒"
    echo "================================================================================"

done

# ================================================================================
# 总结
# ================================================================================
TOTAL_END_TIME=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END_TIME - TOTAL_START_TIME))

echo ""
echo "################################################################################"
echo "全部实验完成！总耗时: ${TOTAL_ELAPSED} 秒"
echo ""
printf "%-15s | %20s | %20s\n" "数据集" "属性+门控 (秒)" "仅属性 (秒)"
echo "----------------|----------------------|----------------------"
for DATASET in "${DATASETS[@]}"; do
    printf "%-15s | %20s | %20s\n" "$DATASET" "${DS_EXP1_TIME[$DATASET]}" "${DS_EXP2_TIME[$DATASET]}"
done
echo ""
echo "结果目录:"
for DATASET in "${DATASETS[@]}"; do
    echo "  [$DATASET] 属性+门控: memory_${DATASET}_attr_gate/  log_${DATASET}_attr_gate/"
    echo "  [$DATASET] 仅属性:    memory_${DATASET}_attr/       log_${DATASET}_attr/"
done
echo "################################################################################"