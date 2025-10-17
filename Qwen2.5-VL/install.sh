conda create -n qwen python=3.10
conda activate qwen
pip install -r requirements_web_demo.txt
git clone --branch v4.51-release --single-branch https://github.com/huggingface/transformers.git
pip install ./transformers
pip install matplotlib==3.7.3 scikit-image scipy

# Pick your CUDA toolkit dir that matches PyTorch's cu121
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH