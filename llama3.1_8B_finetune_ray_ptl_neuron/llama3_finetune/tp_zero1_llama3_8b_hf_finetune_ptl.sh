#!/bin/bash

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training"
export NEURON_FUSE_SOFTMAX=1
# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
# Limit Number of NEFFs
export NEURON_NUM_RECENT_MODELS_TO_KEEP=4
# HOST OOM
export MALLOC_ARENA_MAX=64
#TP degree
TP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
#USE_ZERO_1=1
USE_ZERO_1=1
# global batch size
#: ${GBS:=1024}
GBS=32
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=100
# number of epochs to run
TOTAL_EPOCHS=5
# warmup steps
WARMUP_STEPS=20
# learning rate
LR=3.0e-4
# model path (local copy of Hugging Face llama3.1-8B weights and config files created by downlaod_llama_and_dolly.py)
MODEL_PATH="/shared/trn1_llama_kuberay/hf_llama3.1-8b"
# sequence length
SEQ_LEN=8192
# Path to dataset
DATA_PATH="/shared/trn1_llama_kuberay/databricks/databricks-dolly-15k"
# MODEL_ID
MODEL_ID="NousResearch/Meta-Llama-3.1-8B"
# Checkpoint Dir (for checkpoint converted for use with Trainium)
CKPT_DIR="/shared/trn1_llama_kuberay/Meta-Llama-3.1-8B/"
#############################################

export NUM_NEURONCORES=32
NUM_NODES=2
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
echo "NUM_NODES=$NUM_NODES"

sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

export NEURON_RT_NUM_CORES=32
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

DP=$(($NEURON_RT_NUM_CORES * $NUM_NODES / $TP_DEGREE))
ACC_STEPS=$(($GBS / $MBS / $DP))
#ACC_STEPS=16

if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_exe-$NODE_ID.log
    # EXTRA_ARGS+=" --do_eval"  # disable eval as it currently relies on older optimum-neuron which we aren't installing as part of this tutorial
fi

echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo TOTAL_EPOCHS=$TOTAL_EPOCHS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo SEQ_LEN=$SEQ_LEN

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

python \
    ray_train_llama3.py \
    --model_path $MODEL_PATH \
    --model_id $MODEL_ID \
    --data_dir $DATA_PATH \
    --task "open_qa" \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --num_train_epochs $TOTAL_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --sequence_parallel_enabled \
    --selective_checkpoint_enabled \
    --num_nodes $NUM_NODES \
    --pretrained_ckpt $CKPT_DIR \
    $EXTRA_ARGS |& tee $OUTPUT_LOG
exit ${PIPESTATUS[0]}
