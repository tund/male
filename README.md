# male: MAchine LEarning

## Step 1: Clone *male*

- First, clone *base* repository using `git` with HTTPS URL (recommended):
```sh
git clone https://github.com/dascimal-org/base.git
```
or using `git` via SSH if prefer:
```sh
git clone git@github.com:dascimal-org/base.git
```
- When finish, you will have *base* directory which includes *male* and *tests* directories together with other config files. Just for convinience, rename *base* directory to *male* directory (keep note the address of this *male* directory for Step 2):
```sh
mv base male
```
**Important**: Do not use `python setup.py install` at the moment.

## Step 2: Download and install Anaconda 3 
- Link: `https://www.anaconda.com/download/`. In Linux:
```sh
bash Anaconda3-5.0.1-Linux-x86_64.sh
```
For Window, it's an executable file. So just run it and follow installation steps.
- Don't worry about Python version. You can create Conda environment for any Python version later.
- To use `conda` command in terminal, you must add Anaconda directory to `$PATH` environment variable. It can be selected in installation step or by:
```sh
export PATH="$HOME/anaconda3/bin:$PATH" #$HOME/anaconda3/bin is Anaconda directory
```
In Window, it's better to use Anaconda Prompt. You can add it to `$PATH` if you still insist:
```sh
SET PATH=%PATH%;C:\ProgramData\Anaconda3
```

- Update Anaconda:
```sh
conda update conda
conda update anaconda
```
- Install some important libraries:
```sh
conda install dill scipy scikit-learn matplotlib pillow
```
**Important**: Try to avoid using *pip* (use *conda* if possible)

- To import *male* in Python, you must add *male* directory (in step 1) to $PYTHONPATH
```sh
export PYTHONPATH="$PYTHONPATH:$HOME/male" #Linux
SET PYTHONPATH=%PYTHONPATH%;C:\Users\[username]\male #Window
```
You can try `import male` in Python to see the result.

## Step 3: Install Tensorflow

- The most flexible way to install Tensorflow is installing under created Conda environment. You can find yourself other ways in `https://www.tensorflow.org/install/`. These following steps are for Conda environment way.
- First, create Conda environment (Tensorflow only supports Python version 3.5.x and 3.6.x in Window)
```sh
conda create -n tensorflow36 python=3.6 #Name: tensorflow36, Python version: 3.6
```

- Then, activate this environment (each time you'd like to use Tensorflow)
```sh
activate tensorflow36 #Window
source activate tensorflow36 #Linux
```

- Finally, install Tensorflow: For Linux, use this command (with tfBinaryURL is the URL for your chosen Tensorflow package from: https://www.tensorflow.org/versions/r1.2/install/install_linux#the_url_of_the_tensorflow_python_package) 
**Important**: (For PRaDA students only) Because devcube server is using CuCNN v5, you must install tensorflow v1.2 if you'd like to use GPU.
```sh
#pip install --ignore-installed --upgrade tfBinaryURL
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.2.1-cp36-cp36m-linux_x86_64.whl #example of tensorflow 1.2.1 for Python 3.6 GPU
```
For Windows, use these commands:
```sh
pip install --ignore-installed --upgrade tensorflow  #CPU version
pip install --ignore-installed --upgrade tensorflow-gpu #GPU version
```
You can try `import tensorflow` in Python to see the result.

## Step 4: Test *male*

- `cd` to *male/tests* directory (in step 1) and run this command:
```sh
pytest
```

## Note for GPU version of Tensorflow
- If you install GPU version of Tensorflow, you can get the error about missing some CUDA libraries. For example: missing `libcudart.so.8.0`.
- To solve this, locate if you have this missing library in your machine. 
```sh
locate libcudart.so.8.0 #Example output: /usr/local/cuda-8.0/lib64/libcudart.so.8.0
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-8.0/lib64" #add this directory to LD_LIBRARY_PATH)
```

If not, try install CUDA from NVIDIA first.

------------------
