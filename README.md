# Introduction
This is an updated (compatible with Python 3) fork of the NerveNet paper: [NerveNet: Learning Structured Policy with Graph Neural Networks](http://www.cs.toronto.edu/~tingwuwang/nervenet.html).

# Installation

The original repo uses TF 1.0.1, which is not compatible with Python 3.7. The lowest compatible version is 1.13.1, thus I suggest installing that. GPU version is not necessary.
```bash
pip install tensorflow-gpu==1.13.1
```
If you encounter `TypeError: unsupported operand type(s) for -: 'NoneType' and 'NoneType'`, refer to this issue: https://github.com/WilsonWangTHU/NerveNet/issues/4. Updating the implementation of `_GatherDropNegatives` from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/math_grad.py should fix it.
```bash
pip install gym==0.7.4
pip install 'gym[mujoco]'
```
The following dependencies may be required:
```bash
apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
```
We still rely on **MJPro 1.31**, so we'll need an activation key.

Download `mjpro131` from: https://www.roboti.us/download.html

Download the activation key from: https://www.roboti.us/file/mjkey.txt

Put both in your `.mujoco` folder, and finally install the python bindings: mujoco-py==0.5.7
```bash
pip install mujoco-py==0.5.7
```
Finally, install remaining dependencies:
```bash
pip six beautifulsoup4 termcolor num2words
```
# Running NerveNet
To run the code, first cd into the 'tool' directory.
We provide three examples below (The checkpoint files are already included in the repo):

To test the transfer learning result of **MLPAA** from *centipedeSix* to *centipedeEight*:
```bash
python main.py --task CentipedeEight-v1 --use_gnn_as_policy 0 --num_threads 4 --ckpt_name ../checkpoint/centipede/fc/6 --mlp_raw_transfer 1 --transfer_env CentipedeSix2CentipedeEight  --test 100
```
You should get the average reward around *20*. If you want to test the performance of pretrained models, you should use:
```bash
python main.py --task CentipedeSix-v1 --use_gnn_as_policy 0 --num_threads 4 --ckpt_name ../checkpoint/centipede/fc/6 --mlp_raw_transfer 1  --test 100
```
The performance of the pretrained model of **MLPAA** is around *2755*.

Similarly, to get the transfer learning result of **NerveNet** from *centipedeSix* to *centipedeEight*:
```bash
python main.py --task CentipedeEight-v1 --use_gnn_as_policy 1 --num_threads 4 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB --ckpt_name ../checkpoint/centipede/gnn/6 --transfer_env CentipedeSix2CentipedeEight --test 100
```
The reward of **NerveNet** should be around *1600*. And to test the pretrained model:
```bash
python main.py --task CentipedeSix-v1 --use_gnn_as_policy 1 --num_threads 4 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB --ckpt_name ../checkpoint/centipede/gnn/6 --test 100
```
The reward for **NerveNet** pretrained model is around: *2477*

To train an agent from sratch using NerveNet, you could use the following code:
```bash
python main.py --task ReacherOne-v1 --use_gnn_as_policy 1 --network_shape 64,64 --lr 0.0003 --num_threads 4 --lr_schedule adaptive --max_timesteps 1000000 --use_gnn_as_value 0 --gnn_embedding_option noninput_shared --root_connection_option nN,Rn,uE --gnn_node_option nG,nB
```
