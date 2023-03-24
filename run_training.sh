PORT=$((1024 + RANDOM % 49152))
OUTPUT_DIR=output

TOTAL_BATCH_SIZE=128
NUM_GPUS=4
BATCH_SIZE=4
ACC_STEPS=$((TOTAL_BATCH_SIZE / (NUM_GPUS * BATCH_SIZE)))

MODEL_PATH=facebook/opt-6.7b
MODEL_PATH=/home/groups/deho/jniklaus/LegalLM/llama_7B/llama-7b

if [[ $MODEL_PATH == *"opt"* ]]; then
    TRANSFORMER_LAYER="OPTDecoderLayer"
else
    TRANSFORMER_LAYER="LLaMADecoderLayer"
fi

torchrun --nproc_per_node=${NUM_GPUS} --master_port=${PORT} train.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ./alpaca_data_cleaned.json \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
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
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "${TRANSFORMER_LAYER}"