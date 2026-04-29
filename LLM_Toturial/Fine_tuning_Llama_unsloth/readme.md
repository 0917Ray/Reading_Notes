# How to prepare all the environment:
## 1. Create a new environment
`conda create -n env_name python=3.xx`
(e.g. conda create -n llama_3_env python=3.11)

## 2. Activate the environment
`conda activate env_name`

## 3. Install PyTorch
for CUDA 12.8: `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128`

for CUDA 13.0: `pip3 install torch torchvision`

## 4. Install Unsloth
`pip install unsloth`

## 5. Install other required packages
`pip install -r requirements.txt`
(Note: you can use `pwd` and `cd your_file_name` to make sure you are in the right place)

# How to use this tutorial
## Way 1
Run `test_demo` step by step

## Way 2
1. Run `llama_70B_tuning.py` to get the fine-tuning results (logs and figures)
2. Run `llama_70B_inference.py` to test the model's ability to response prompt
