# neural-style-transfer
[neural style transfer](https://arxiv.org/pdf/1508.06576) paper implemented using [tinygrad](https://github.com/tinygrad/tinygrad/tree/master)  

- pip install -r requirements.txt
- DEBUG=2 IMAGE=1 FLOAT16=1 BEAM=1 python nst.py

Notes:  
- BEAM search takes longer than I anticipated, so start with 1 or 2 and go higher. 
- tried with both CUDA and OpenCL, runs well in both cases but idk if I am using all the tinygrad optimizations correctly, JIT seems to do something tho
- set the right CUDA_PATH, if you see "RuntimeError: module load failed with status code CUDA_ERROR_UNSUPPORTED_PTX_VERSION: CUDA_ERROR_UNSUPPORTED_PTX_VERSION" and update drivers. 


