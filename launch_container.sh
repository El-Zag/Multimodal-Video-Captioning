DATA_DIR=$1
MODEL_DIR=$2
OUTPUT=$3
RAW=$4
PICKLES=$5
CUDA_VISIBLE_DEVICES=$6
https_proxy=http://127.0.0.1:3128
http_proxy=http://127.0.0.1:3128


docker run --gpus '"'device=$CUDA_VISIBLE_DEVICES'"' --ipc=host --rm -it \
    --network=host --env http_proxy=$http_proxy --env https_proxy=$https_proxy \
    --mount src=$(pwd),dst=/videocap,type=bind \
    --mount src=$DATA_DIR,dst=/videocap/datasets,type=bind\
    --mount src=$MODEL_DIR,dst=/videocap/models,type=bind \
    --mount src=$OUTPUT,dst=/videocap/output,type=bind \
    --mount src=$RAW,dst=/videocap/rawvideos,type=bind \
    --mount src=$PICKLES,dst=/videocap/pickles,type=bind \
    -e NVIDIA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
    -w /videocap linjieli222/videocap_torch1.7:fairscale \
    bash -c "source /videocap/setup.sh && bash" 
