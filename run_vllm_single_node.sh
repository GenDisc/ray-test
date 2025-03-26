#!/bin/bash

MODEL_LIST=(
  "blip"
  "llava"
  "llama"
)

MODEL_FULL_NAME=(
  "Salesforce/blip2-opt-2.7b ./template/template_blip2.jinja"
  "llava-hf/llava-1.5-7b-hf ./template/template_llava.jinja"
  "meta-llama/Llama-3.2-11B-Vision"
)

check_model() {
  local key="$1"
  for i in "${!MODEL_LIST[@]}"; do
    if [ "${MODEL_LIST[$i]}" = "$key" ]; then
      echo "${MODEL_FULL_NAME[$i]}" # Return the associated value
      return 0
    fi
  done
  echo "Model '$key' not found." # Key does not exist
  return 1
}

install_dependencies() {
  echo "Installing Dependencies..."
  export NCCL_P2P_DISABLE=1
}

start_ray() {
  if [ "$NODE_RANK" == 0 ]; then
    poetry run ray metrics launch-prometheus
    poetry run ray start --head --port=6379 --metrics-export-port=8080 --dashboard-host='0.0.0.0' &
    sleep 20
  else
    sleep 5
    poetry run ray start --address="$MASTER_ADDR:6379" --dashboard-host='0.0.0.0' &
    sleep infinity
  fi
}

check_ray() {
  python3 <<EOF
import os
import ray
import time
ray.init()

WORLD_SIZE = int(os.environ['WORLD_SIZE'])
node_rank = int(os.environ['NODE_RANK'])
print('WORLD_SIZE: %d' % WORLD_SIZE)

if node_rank == 0:
    while True:
        if int(ray.cluster_resources()['GPU']) == WORLD_SIZE:
            print('''This cluster consists of
                {} nodes in total
                {} GPU resources in total
            '''.format(len(ray.nodes()), ray.cluster_resources()['GPU']))
            exit()
        time.sleep(3)
else:
    exit()

EOF
}

test() {
  read model_name template <<<"$1"
  echo "$model_name"
  if [ -z "$template" ]; then
    echo "String is empty"
  else
    echo "$template"
  fi
}

deploy_vllm() {
  read model_name template <<<"$1"
  echo "$model_name"
  echo "running with existing template: $template"
  if [ -z "$template" ]; then
    poetry run serve run ray_deployment:build_app \
      accelerator="GPU" \
      limit-mm-per-prompt='image=1' \
      max_model_len=1200 \
      max_num_seqs=16 \
      model="$model_name"
  else
    poetry run serve run ray_deployment:build_app \
      accelerator="GPU" \
      limit-mm-per-prompt='image=1' \
      max_model_len=1200 \
      max_num_seqs=16 \
      model="$model_name" \
      chat_template="$template"
  fi
}

main() {
  if [ -z "$1" ]; then
    echo "error: A Model name must be given.\nUsage: $0 <value>"
    exit 1
  fi
  model_meta=$(check_model "$1")
  deploy_vllm "$model_meta"
}

main "$1"
