#!/usr/bin/env bash
set -uxo pipefail
export VERL_HOME=${VERL_HOME:-"/data/xiaochangyi/DAPO_verl"}
export TRAIN_FILE=${TRAIN_FILE:-"${VERL_HOME}/data/dapo-math-17k.parquet"}
export TEST_FILE=${TEST_FILE:-"${VERL_HOME}/data/MATH-500.parquet"}

mkdir -p "${VERL_HOME}/data"

wget -O "${TRAIN_FILE}" "https://hf-mirror.com/datasets/BytedTsinghua-SIA/DAPO-Math-17k/resolve/main/data/dapo-math-17k.parquet?download=true"

wget -O "${TEST_FILE}" "https://hf-mirror.com/datasets/HuggingFaceH4/MATH-500/blob/main/test.jsonl"