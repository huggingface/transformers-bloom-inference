export TOKENIZERS_PARALLELISM=false
export MODEL_NAME=bigscience/bloom
export DEPLOYMENT_FRAMEWORK=hf_accelerate
export DTYPE=fp16
export MAX_INPUT_LENGTH=2048
export NUM_GPUS=8

# for more information on gunicorn see https://docs.gunicorn.org/en/stable/settings.html
gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'
