#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python estimate_aRF.py 21067-10-18
sleep 5s

CUDA_VISIBLE_DEVICES=1 python estimate_aRF.py 23343-5-17
sleep 5s

CUDA_VISIBLE_DEVICES=2 python estimate_aRF.py 22846-10-16
sleep 5s

CUDA_VISIBLE_DEVICES=3 python estimate_aRF.py 23656-14-22
sleep 5s

CUDA_VISIBLE_DEVICES=5 python estimate_aRF.py 23964-4-22
wait
