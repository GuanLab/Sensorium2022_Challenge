# **Predicting and Interpreting the single neuron response of the visual cortex via Deep Learning Model**

This is the package of our winning solution in the SENSORIUM 2022 Challenge. Details of this challenge can be found in [this paper](https://arxiv.org/abs/2206.08666). Please contact ([dengkw@umich.edu](mailto:dengkw@umich.edu) or [gyuanfan@umich.edu](mailto:gyuanfan@umich.edu)) if you have any questions or suggestions.

Overview of the data and experiment design
![Figure1](figs/Figure1.png?raw=true "Title")

Overview of the methods
![Figure2](figs/Figure2.png?raw=true "Title")

---

## Installations

Git clone a copy of code:

```bash
git clone https://github.com/GuanLab/Sensorium2022_Challenge.git
```

Setup the running environment through conda

```bash
conda create env -f environment.yml
conda activate sensorium
```

Install the YOLOv5

```bash
# you may need to ensure the directory name is "yolov5"
git clone https://github.com/ultralytics/yolov5.git
```

Install the data under `dataset` directory: **[https://gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022)**

The pretrained weights can be retrieved from **[google drive](https://gin.g-node.org/cajal/Sensorium2022)**. Save them under the `model_checkpoints` folder

## Build the model on the challenge data

#### Data pre-processing

After downloading and unzipping the challenge data, follow the scripts in `0_process_data.ipynb` to label the bounding boxes and generate different train-validation splits for ensemble. The `yolov5l.pt` is the official pretrained weights downloaded [here](https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt), and the `yolo-finetune.pt` is our fine-tuned weights on [ILSVRC2017](https://www.kaggle.com/c/imagenet-object-localization-challenge).

#### Train and evaluate model

Follow the scripts in `sensorium/1_train_evaluate_submit.ipynb`, you will be able to train and evaluate the model on the challenge data, and repeat the performance reported in our paper.

1. Training

   ```bash
   # You may want to change visible gpus in this script.
   bash run.sh
   ```
2. Predict and evaluate

   ```bash
   CUDA_VISIBLE_DEVICES=0 python predict.py

   # get the performance for each neuron
   CUDA_VISIBLE_DEVICES=0 python predict_per_neuron.py 10
   ```
3. Generate the predictions and corresponding responses (the ground-truths) for analyzing

   ```bash
   CUDA_VISIBLE_DEVICES=0 python submit.py
   ```

#### Analyze the predictions (optional)

We provide the scripts in `analyze` to repeat our results and some of the figures in the paper. They include extracting the image properties (complexity, brightness, contrast) `inspect_model_with_image.ipynb` and analyzing the spatial properties `grid_experiment.ipynb`

## Reference

[https://github.com/sinzlab/neuralpredictors](https://github.com/sinzlab/neuralpredictors)

[https://github.com/sinzlab/sensorium](https://github.com/sinzlab/sensorium)
