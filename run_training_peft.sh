PORT=$((1024 + RANDOM % 49152))
export WANDB_PROJECT=LegalLM
conda activate legallm
huggingface-cli login --token $HUGGINGFACE_TOKEN

TRAIN_WITH_PEFT=True
LEARNING_RATE=2e-4

MAX_SEQ_LEN=512

# Set BATCH_SIZE based on the given MAX_SEQ_LEN (tested on 1 80GB A100 GPUs for a 12B model)
if [ "$MAX_SEQ_LEN" -eq 512 ]; then
  BATCH_SIZE=32
elif [ "$MAX_SEQ_LEN" -eq 1024 ]; then
  BATCH_SIZE=16
elif [ "$MAX_SEQ_LEN" -eq 2048 ]; then
  BATCH_SIZE=8
else
  echo "Invalid MAX_SEQ_LEN value"
  exit 1
fi

# Memory consumption
# 20B model, 2048 seq len, batch size 1: 42GB ==> 103 s/it
# 12B model, 2048 seq len, batch size 1: 26GB
# 12B model, 2048 seq len, batch size 8: 57GB ==> 80s/it
# 12B model, 1024 seq len, batch size 16: 74GB ==> 45s/it
# 12B model, 512 seq len, batch size 32: 66GB ==> 23s/it
# 6.9B model, 2048 seq len, batch size 1: 20GB
# 70M model, 2048 seq len, batch size 1: 10GB


TOTAL_BATCH_SIZE=128
export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) # get number of available GPUs
NUM_GPUS=1
ACC_STEPS=$((TOTAL_BATCH_SIZE / (NUM_GPUS * BATCH_SIZE)))

SAMPLES_PER_DATASET=1000

DATA_PATH=./alpaca_data.json
DATA_PATH=./alpaca_data_cleaned.json
DATA_PATH=./law_instruction_data_len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET}.json # use a fixed dataset if we generated it already
DATA_PATH=max-seq-len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET} # generate the dataset on the fly

# overview: https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4

# instruction tuned models
#MODEL_PATH=stabilityai/stablelm-tuned-alpha-7b


# base models
# 7B scale
#MODEL_PATH=llama-7b
#MODEL_PATH=cerebras/Cerebras-GPT-6.7B # this is apparently not great
#MODEL_PATH=EleutherAI/pythia-6.9b
#MODEL_PATH=stabilityai/stablelm-base-alpha-7b # this is apparently not great
#MODEL_PATH=facebook/opt-6.7b # this is apparently not great

# 12B scale ==> use this scale for now
#MODEL_PATH=llama-13b
MODEL_PATH=EleutherAI/pythia-12b
#MODEL_PATH=cerebras/Cerebras-GPT-13B # this is apparently not great
#MODEL_PATH=facebook/opt-13b # this is apparently not great
#MODEL_PATH=google/mt5-xxl # 13B ==> more complicated because it is MT5ForConditionalGeneration

# 20B scale
#MODEL_PATH=EleutherAI/gpt-neox-20b

# 30B scale
#MODEL_PATH=llama-33b
#MODEL_PATH=facebook/opt-30b # this is apparently not great

# 60B scale
#MODEL_PATH=llama-65b
#MODEL_PATH=facebook/opt-66b # this is apparently not great

# debug models
#MODEL_PATH=cerebras/Cerebras-GPT-111M # for debugging
#MODEL_PATH=EleutherAI/pythia-70m # for debugging
#MODEL_PATH=facebook/opt-125m # for debugging

OUTPUT_DIR=output-${MODEL_PATH##*/}-max-seq-len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET}

python3 train.py \
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
    --save_steps 100 \
    --save_total_limit 3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
