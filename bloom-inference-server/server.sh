export MODEL_NAME=bigscience/bloom
export DEPLOYMENT_FRAMEWORK=hf_accelerate
export DTYPE=fp16

gunicorn -t 0 -w 1 -b 127.0.0.1:5000 server:app
