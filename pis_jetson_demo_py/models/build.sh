#!/bin/bash
#
# ONNX to TensorRT converter script
# @author An Jung-In <jian@fssolution.co.kr>
#
# [Usage]
# 1. Put .onnx file inside onnx/ folder.
#   - ONNX filename should match following format: <model_name>.<type>.onnx,
# 2. Create new .env file on same directory. See .template.env file.
# 3. Run ./build.sh

# Set this to 1 to skip exising engine file
SKIP_EXISING=1

DEFAULT_ENV="./onnx/.default.env"
source ${DEFAULT_ENV}

TRT_VERSION=$(/usr/src/tensorrt/bin/trtexec -h | head -n 1 | grep -oE '[[:digit:]]+')

# function: Run single conversion
run() {
  CACHE_PATH="cache/${BASENAME}_trt${TRT_VERSION}.cache"

  if [ ! -f "${CACHE_PATH}" ]; then
    echo "cache not found, generating (may take a few hours!)"
  fi

  EXTRA_FLAGS=
  HEADING=fp32_
  if [ "$FP16" -eq "1" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --fp16"
    HEADING=fp16_
  fi

  ENGINE_FILE_PATH="engine/${BASENAME}.${TYPE}.${HEADING}b${MIN_BATCHES}_b${MAX_BATCHES}_trt${TRT_VERSION}.engine"

  if [ "${SKIP_EXISING}" -eq "1" ] && [ -f "${ENGINE_FILE_PATH}" ]; then
    echo "Found engine file \"${ENGINE_FILE_PATH}\", skipping ..."
    return 0
  fi

  SHAPE_OPTS="--minShapes=images:${MIN_BATCHES}x3x${INPUT_SIZE} \
      --optShapes=images:${MAX_BATCHES}x3x${INPUT_SIZE} \
      --maxShapes=images:${MAX_BATCHES}x3x${INPUT_SIZE}"

  if [ "${MIN_BATCHES}" -eq "1" ] && [ "${MAX_BATCHES}" -eq "1" ]; then
    SHAPE_OPTS=""
  fi


  echo "Running ${BASENAME}.${TYPE}... (see output.log for details)"
  TIMEFORMAT=%R
  ELAPSED_TIME=$(
    TIMEFORMAT="%1R"
    { time /usr/src/tensorrt/bin/trtexec \
      --onnx="onnx/${BASENAME}.${TYPE}.onnx" \
      ${EXTRA_FLAGS} --workspace=4096 \
      --saveEngine=${ENGINE_FILE_PATH} \
      ${SHAPE_OPTS} \
      --timingCacheFile="${CACHE_PATH}" \
      2>&1 | tee -a output.log >/dev/null; } 2>&1
  )
  echo "Done in ${ELAPSED_TIME}sec. Exported engine path:"
  echo "${ENGINE_FILE_PATH}"
  echo ""
}

# function: Main script
main() {
  CONFIG_ENVFILES=$(ls -1 onnx/*.env | sort -V)

  # Sanity check
  for ENV_FILENAME in ${CONFIG_ENVFILES[@]}; do
    source ${DEFAULT_ENV}
    source ${ENV_FILENAME}
    ONNX_FILENAME="./onnx/${BASENAME}.${TYPE}.onnx"
    if [ ! -f "${ONNX_FILENAME}" ]; then
      echo "fatal: Cannot find onnx model \"${ONNX_FILENAME}\" with configuration file \"${ENV_FILENAME}\"."
      return 1
    fi
  done

  for ENV_FILENAME in ${CONFIG_ENVFILES[@]}; do
    source ${DEFAULT_ENV}
    source ${ENV_FILENAME}

    echo "============= Configuration ============="
    echo "- BASENAME=${BASENAME}"
    echo "- TYPE=${TYPE}"
    echo "- MIN_BATCHES=${MIN_BATCHES}"
    echo "- MAX_BATCHES=${MAX_BATCHES}"
    echo "- INPUT_SIZE=${INPUT_SIZE}"
    echo "========================================="
    run
  done
}

main
