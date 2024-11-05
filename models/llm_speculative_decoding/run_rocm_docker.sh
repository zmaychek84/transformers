#!/bin/bash

HERE=$(pwd -P) # Absolute path of current directory
user=$(whoami)
uid=$(id -u)
gid=$(id -g)

docker run \
  -v /dev/shm:/dev/shm \
  -v /opt/xilinx/dsa:/opt/xilinx/dsa \
  -v /opt/xilinx/overlaybins:/opt/xilinx/overlaybins \
  -v /group/dphi_algo_scratch_17/zepingl/:/group/dphi_algo_scratch_17/zepingl/ \
  -v /group/modelzoo/sequence_learning/:/group/modelzoo/sequence_learning/ \
  -v /scratch1_nvme_1/workspace/:/scratch1_nvme_1/workspace/ \
  -e USER=$USER -e UID=$uid -e GID=$gid \
  -v $HERE:/group/dphi_algo_scratch_17/zepingl/work/verify/spec_dec_simplest/ \
  -w /group/dphi_algo_scratch_17/zepingl/work/verify/spec_dec_simplest/ \
  --device=/dev/kfd \
  --device=/dev/dri \
  --security-opt seccomp=unconfined \
  -it \
  --network=host \
  --name="spec-dec-simplest" \
  bbf87cd80687   ##local docker image
  # xcoartifactory.xilinx.com/uif-docker-master-local/rocmiv-internal-pt:_ubuntu20.04_py3.8_pytorch_release-1.13_3aa2ef3
  # xcoartifactory.xilinx.com/uif-docker-release-local/uif1.2/release/uif-pytorch:5.6_93_vai_3.5_py3.8_pytorch1.13_bee46f0
