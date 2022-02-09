#!/bin/bash -l

#SBATCH -J  CQA_v1.0           # --job-name=singularity
#SBATCH -o  CQA_v1.0.%j.out     # slurm output file (%j:expands to %jobId)
#SBATCH -p  A100-pci              # queue or partiton name ; sinfo  output
#SBATCH -t  72:00:00          # 작업시간(hh:mm:ss) 1.5 hours 설정
#SBATCH -N  1                 # --nodes=1 (고정)
#SBATCH -n  1                 # --tasks-per-node=1  노드당 할당한 작업수 (고정)
#SBATCH --gres=gpu:1          # gpu Num Devices  가급적이면  1,2,4.6,8  2배수로 합니다.

conda activate chain

module purge
module add cuda/11.0
module add cuDNN/cuda/11.0/8.0.4.30

DOMAIN=wikipedia

EPOCHS=4
BATCH_SIZE=8
LR=1e-5
WARMUP=0.0
NUM_TURN=2

TAG=$DOMAIN

OUTPUT_DIR=output-cmp
RESULT_DIR=$EPOCHS.$BATCH_SIZE.$LR.$WARMUP-$TAG
MODEL_NAME=bert-large-cased

python run_cqa.py --epochs $EPOCHS \
                  --batch-size $BATCH_SIZE \
                  --learning-rate $LR \
                  --warmup-prop $WARMUP \
                  --output-dir $OUTPUT_DIR \
                  --result-dir $RESULT_DIR \
                  --num-turn $NUM_TURN \
                  --model-name $MODEL_NAME \
                  --coqa-domain $DOMAIN 
                       
python run_cqa_inference.py --batch-size 4 \
                            --output-dir $OUTPUT_DIR \
                            --result-dir $RESULT_DIR \
                            --num-turn $NUM_TURN \
                            --model-name $MODEL_NAME \
                            --coqa-domain $DOMAIN 

python tool/convert_coqa.py \
  --input_file=$OUTPUT_DIR/$RESULT_DIR/predictions.json \
  --output_file=$OUTPUT_DIR/$RESULT_DIR/eval_form.json

python tool/eval_coqa.py \
  --data-file=data/coqa/coqa-dev-$DOMAIN.json \
  --pred-file=$OUTPUT_DIR/$RESULT_DIR/eval_form.json \
  >> $OUTPUT_DIR/$RESULT_DIR/performance.json                             
