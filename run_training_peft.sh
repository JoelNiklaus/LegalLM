PORT=$((1024 + RANDOM % 49152))
export WANDB_PROJECT=LegalLM
conda activate legallm
huggingface-cli login --token $HUGGINGFACE_TOKEN

TRAIN_WITH_PEFT=True
LEARNING_RATE=2e-4

# IMPORTANT make sure to use bf16 if model was trained with bf16, if trained with fp16, use fp16
# TODO try training in bf16 and NOT loading in 8 bit and disable tf32
# pythia models: fp16
# https://github.com/huggingface/transformers/issues/20287

# TODO maybe give lora more capacity when training on more datasets ==> probably not it because alpaca-lora had even lower capacity than we do

# TODO maybe we need more examples per dataset to get better results (casehold has 45K and we only train on 1K) ==> 10K did not change results either

# TODO maybe we need to bring more diversity into the multiple choice answer styles: not always (a), (b) ==> make this more diverse

MAX_SEQ_LEN=512

export CUDA_VISIBLE_DEVICES=0 # restrict usable GPUs here
#NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l) # does not work with CUDA_VISIBLE_DEVICES
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l | tr -d ' ')
echo "Using $NUM_GPUS GPUs"

if [ "$NUM_GPUS" -eq 1 ]; then
    # Set BATCH_SIZE based on the given MAX_SEQ_LEN (tested on 1 80GB A100 GPUs for a 12B model) or was it a 7B model???
    if [ "$MAX_SEQ_LEN" -eq 512 ]; then
      BATCH_SIZE=6
    elif [ "$MAX_SEQ_LEN" -eq 1024 ]; then
      BATCH_SIZE=2
    elif [ "$MAX_SEQ_LEN" -eq 2048 ]; then
      BATCH_SIZE=1
    else
      echo "Invalid MAX_SEQ_LEN value"
      exit 1
    fi
elif [ "$NUM_GPUS" -eq 2 ]; then
    # Set BATCH_SIZE based on the given MAX_SEQ_LEN (tested on 2 80GB A100 GPUs for a 7B model)
    if [ "$MAX_SEQ_LEN" -eq 512 ]; then
      BATCH_SIZE=16
    elif [ "$MAX_SEQ_LEN" -eq 1024 ]; then
      BATCH_SIZE=8
    elif [ "$MAX_SEQ_LEN" -eq 2048 ]; then
      BATCH_SIZE=2
    else
      echo "Invalid MAX_SEQ_LEN value"
      exit 1
    fi
else
  echo "Invalid NUM_GPUS value"
  exit 1
fi

GRADIENT_CHECKPOINTING=False
if [ "$GRADIENT_CHECKPOINTING" = True ]; then
    # multiply the batch size by 8
    BATCH_SIZE=$((BATCH_SIZE * 8))
fi

echo "Using batch size $BATCH_SIZE"


# Memory consumption
# 20B gpt neox model, 2048 seq len, batch size 1: 42GB ==> 206 s/it
# 12B pythia model, 2048 seq len, batch size 1: 26GB
# 12B pythia model, 2048 seq len, batch size 8: 57GB ==> 160s/it
# 12B pythia model, 1024 seq len, batch size 16: 74GB ==> 90s/it
# 12B pythia model, 512 seq len, batch size 32: 66GB ==> 46s/it
# 6.9B pythia model, 512 seq len, batch size 128: 55GB & 79GB ==> 31s/it
# 7B RedPajama model, 512 seq len, batch size 128: 55GB & 79GB ==> 31s/it
# 7B RedPajama model, 512 seq len, batch size 16 (no grad checkpoint: 42GB & 60GB ==> 21s/it
# 7B RedPajama model, 512 seq len, batch size 2 (no grad checkpoint: 74GB & 60GB ==> 21s/it


TOTAL_BATCH_SIZE=256
ACC_STEPS=$((TOTAL_BATCH_SIZE / (NUM_GPUS * BATCH_SIZE)))

SAMPLES_PER_DATASET=10000

DATA_PATH=./alpaca_data.json
DATA_PATH=./alpaca_data_cleaned.json
DATA_PATH=./law_instruction_data_len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET}.json # use a fixed dataset if we generated it already
DATA_PATH=max-seq-len:${MAX_SEQ_LEN}_samples:${SAMPLES_PER_DATASET} # generate the dataset on the fly

# overview: https://docs.google.com/spreadsheets/d/1kT4or6b0Fedd-W_jMwYpb63e1ZR3aePczz3zlbJW-Y4

# instruction tuned models
#MODEL_PATH=stabilityai/stablelm-tuned-alpha-7b
#MODEL_PATH=mosaicml/mpt-7b-instruct
#MODEL_PATH=togethercomputer/RedPajama-INCITE-Instruct-7B-v0.1


# base models
# 7B scale
#MODEL_PATH=llama-7b
#MODEL_PATH=cerebras/Cerebras-GPT-6.7B # this is apparently not great
MODEL_PATH=EleutherAI/pythia-6.9b
#MODEL_PATH=stabilityai/stablelm-base-alpha-7b # this is apparently not great
#MODEL_PATH=facebook/opt-6.7b # this is apparently not great

# 12B scale ==> use this scale for now
#MODEL_PATH=llama-13b
#MODEL_PATH=EleutherAI/pythia-12b
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

HF_NAME="lawinstruct/LegalLM-$(basename ${MODEL_PATH})-lora"

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
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 20 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --fp16 True \

#    --bf16 True \
#    --tf32 True \

# does not work yet on apex machine because of git lfs
    #--hub_model_id ${HF_NAME} \
    #--hub_strategy=checkpoint \
    #--push_to_hub \
    #--hub_private_repo \
    #--hub_token ${HUGGINGFACE_TOKEN} \
