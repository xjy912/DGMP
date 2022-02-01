# Directed graph convolution network with multi-layer perceptron

## Requirements

Our project is developed using Python 3.7
PyTorch 1.5.0 with CUDA10.2. 
We recommend you to use [anaconda](https://www.anaconda.com/) for dependency configuration.

First create an anaconda environment called ```DGMP``` by

```shell
conda create -n DGMP python=3.7
conda activate DGMP
```

Then, you need to install torch manually to fit in with your server environment (e.g. CUDA version). run

```shell
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
```

Besides, torch-scatter and torch-sparse are required for dealing with sparse graph. 
For these two packages, please follow their official instruction [torch-scatter](https://github.com/rusty1s/pytorch_scatter) and [torch-sparse](https://github.com/rusty1s/pytorch_sparse).

```shell
cd DGMP
pip install torch-geometric
pip install torch-scatter 
```
## Run

```shell
cd code
python gcn.py --gpu-no 0 --dataset cancer
python DGMP.py --gpu-no 0 --dataset cancer
python MLP.py --gpu-no 0 --dataset cancer

cpu
python gcn.py --gpu-no -1 --dataset cancer
python DGMP.py --gpu-no -1 --dataset cancer
python MLP.py --gpu-no -1 --dataset cancer
```

## License

DGMP is released under the MIT License. See the LICENSE file for more details.

## Acknowledgements

We thank the authors of following works for opening source their excellent codes.[DiGCN](https://github.com/flyingtango/DiGC)

