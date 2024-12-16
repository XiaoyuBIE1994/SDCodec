## 1. Install Bazel

Here is an example, you may need to change the download link to the latest version, and adapt to your conda env filepath

```bash
# Linux for example

# Check the architecture of your computer/cluster
dpkg --print-architecture

# Download the right version of Bazelisk binary realease, e.g. linux-amd64
# check the latest version: https://github.com/bazelbuild/bazelisk/releases
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.19.0/bazelisk-linux-amd64

# Rename and move to your conda env, e.g. torch2
chmod +x bazelisk-linux-amd64
mv bazelisk-linux-amd64 /home/ids/xbie/anaconda3/envs/torch2/bin/bazel

# Now you can run bazel by activating the conda env
```

## 2. Install VisQOL
```bash
# Clone the VisQOL repo
git clone https://github.com/google/visqol.git

# Install bin, need GCC 12
cd visqol
bazel build :visqol -c opt  --jobs=10

# Install Python API
pip install .
```

## 3. Potential Problem
Here are some problems I have encountered during installation and how I solved them 
```bash
# It might encounter gcc/g++ compilation problem, even if you have successfully build visqol
# e.g. libstdc++.so.6: version `GLIBCXX_3.4.30' not found (required by /home/xbie/anaconda3/envs/torch2/lib/python3.11/site-packages/visqol/visqol_lib_py.so)
# in that case, masure your gcc/g++ version is >= 12
conda install conda-forge::gxx_impl_linux-64
conda install -c conda-forge gcc=12.1.0

# In case your Bazel is installed in the system path
# then you should update gcc/g++ in system-wise
sudo apt install --reinstall gcc-12
sudo apt install --reinstall g++-12
sudo ln -s -f /usr/bin/gcc-12 /usr/bin/gcc
sudo ln -s -f /usr/bin/g++-12 /usr/bin/g++

# In case fail to import `visqol_lib_py` with unkown path
# install it manually, make sure you are in the Visqol root path
# if on the cluster, use `--jobs=10` to limit the parallel jobs
# then we need to re-run the installation to link the .so library
bazel build -c opt //python:visqol_lib_py.so --jobs=10
pip install .

# If still have problems,
# remove all Bazel cache files and re-run the installation
rm -rf ~/.cache/bazel
rm -rf ~/.cache/bazelisk
``
