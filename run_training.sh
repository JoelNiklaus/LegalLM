PORT=$((1024 + RANDOM % 49152))
OUTPUT_DIR=output

TOTAL_BATCH_SIZE=128
NUM_GPUS=2
BATCH_SIZE=4
ACC_STEPS=$((TOTAL_BATCH_SIZE / (NUM_GPUS * BATCH_SIZE)))
MAX_SEQ_LEN=512

# Maximum Sequence Lengths and batch sizes:
# 512: 4
# 1024: 3
# 2048: 2

DATA_PATH=./alpaca_data_cleaned.json

MODEL_PATH=facebook/opt-6.7b
#MODEL_PATH=/home/groups/deho/jniklaus/LegalLM/llama-7b
MODEL_PATH=/home/groups/deho/jniklaus/LegalLM/llama_7B/llama-7b
MODEL_PATH=cerebras/Cerebras-GPT-6.7B

if [[ $MODEL_PATH == *"opt"* ]]; then
    TRANSFORMER_LAYER="OPTDecoderLayer"
elif [[ $MODEL_PATH == *"llama"* ]]; then
    TRANSFORMER_LAYER="LLaMADecoderLayer"
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
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --bf16 True \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "${TRANSFORMER_LAYER}"

