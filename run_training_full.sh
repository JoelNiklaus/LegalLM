# IMPORTANT: It seems like LLaMA needs 4 GPUs and does not run on 2 GPUs with this batch size.

PORT=$((1024 + RANDOM % 49152))

TRAIN_WITH_PEFT=False
LEARNING_RATE=2e-5

MAX_SEQ_LEN=512

# Set BATCH_SIZE based on the given MAX_SEQ_LEN (tested on 2 80GB A100 GPUs)
if [ "$MAX_SEQ_LEN" -eq 512 ]; then
  BATCH_SIZE=4
elif [ "$MAX_SEQ_LEN" -eq 1024 ]; then
  BATCH_SIZE=1 # cerebras-gpt-6.7b can do 2
elif [ "$MAX_SEQ_LEN" -eq 1536 ]; then
  BATCH_SIZE=1 # does not work for stablelm and probably not for llama
else
  echo "Invalid MAX_SEQ_LEN value"
  exit 1
fi

TOTAL_BATCH_SIZE=128
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) # get number of available GPUs
ACC_STEPS=$((TOTAL_BATCH_SIZE / (NUM_GPUS * BATCH_SIZE)))

CUDA_VISIBLE_DEVICES=2,3
NUM_GPUS=2

SAMPLES_PER_DATASET=10

DATA_PATH=./alpaca_data.json
DATA_PATH=./law_instruction_data_len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET}.json # use a fixed dataset if we generated it already
DATA_PATH=./alpaca_data_cleaned.json
DATA_PATH=max-seq-len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET} # generate the dataset on the fly

MODEL_PATH=cerebras/Cerebras-GPT-111M # for debugging
MODEL_PATH=stabilityai/stablelm-base-alpha-7b # this is apparently not great
MODEL_PATH=/home/groups/deho/jniklaus/LegalLM/llama_7B/llama-7b
MODEL_PATH=cerebras/Cerebras-GPT-6.7B # this is apparently not great
MODEL_PATH=facebook/opt-6.7b # this is apparently not great
MODEL_PATH=EleutherAI/pythia-12b
MODEL_PATH=EleutherAI/pythia-6.9b

OUTPUT_DIR=output-${MODEL_PATH##*/}-max-seq-len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET}

if [[ $MODEL_PATH == *"opt"* ]]; then
    TRANSFORMER_LAYER="OPTDecoderLayer"
elif [[ $MODEL_PATH == *"llama"* ]]; then
    TRANSFORMER_LAYER="LLaMADecoderLayer"
elif [[ $MODEL_PATH == *"stablelm"* ]]; then
    TRANSFORMER_LAYER="GPTNeoXLayer"
else
    TRANSFORMER_LAYER="GPT2Block"
fi


torchrun --nproc_per_node=${NUM_GPUS} --master_port=${PORT} train.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --model_max_length ${MAX_SEQ_LEN} \
    --num_train_epochs 3 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --train_with_peft ${TRAIN_WITH_PEFT} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "${TRANSFORMER_LAYER}"

