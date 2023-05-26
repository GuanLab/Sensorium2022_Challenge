
CUDA_VISIBLE_DEVICES=0 python train.py 1 &
CUDA_VISIBLE_DEVICES=0 python train.py 2 &
CUDA_VISIBLE_DEVICES=1 python train.py 3 &
CUDA_VISIBLE_DEVICES=1 python train.py 4 &
CUDA_VISIBLE_DEVICES=2 python train.py 5 &
wait

CUDA_VISIBLE_DEVICES=0 python train.py 6 &
CUDA_VISIBLE_DEVICES=0 python train.py 7 &
CUDA_VISIBLE_DEVICES=1 python train.py 8 &
CUDA_VISIBLE_DEVICES=1 python train.py 9 &
CUDA_VISIBLE_DEVICES=2 python train.py 10 &
wait
