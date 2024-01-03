# We used the following commands to create our environment

conda create -n sensorium2022 python=3.8 mamba
conda activate sensorium2022

mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
mamba install matplotlib scikit-learn pandas seaborn tqdm gitpython datajoint ipykernel scikit-image

pip install neuralpredictors==0.3.0
pip install nnfabrik==0.2.1
pip install lipstick