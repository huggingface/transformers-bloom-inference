export MODEL_NAME="bigscience/bloom"

gunicorn -w 1 -b 127.0.0.1:5000 server:app
