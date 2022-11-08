gen-proto:
	pip install grpcio-tools==1.50.0

	mkdir -p inference_server/model_handler/grpc_utils/pb

	python -m grpc_tools.protoc -Iinference_server/model_handler/grpc_utils/proto --python_out=inference_server/model_handler/grpc_utils/pb --grpc_python_out=inference_server/model_handler/grpc_utils/pb inference_server/model_handler/grpc_utils/proto/generation.proto

	find inference_server/model_handler/grpc_utils/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;

	touch inference_server/model_handler/grpc_utils/__init__.py
	touch inference_server/model_handler/grpc_utils/pb/__init__.py

	rm -rf inference_server/model_handler/grpc_utils/pb/*.py-e

bloom-176b:
	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=microsoft/bloom-deepspeed-inference-fp16 \
	DEPLOYMENT_FRAMEWORK=ds_inference \
	DTYPE=fp16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	NUM_GPUS=8 \
	gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

bloomz-176b:
	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=bigscience/bloomz \
	DEPLOYMENT_FRAMEWORK=ds_inference \
	DTYPE=fp16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	NUM_GPUS=8 \
	gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'

bloomz-560m:
	TOKENIZERS_PARALLELISM=false \
	MODEL_NAME=bigscience/bloom-560m \
	DEPLOYMENT_FRAMEWORK=ds_inference \
	DTYPE=fp16 \
	MAX_INPUT_LENGTH=2048 \
	MAX_BATCH_SIZE=4 \
	NUM_GPUS=8 \
	gunicorn -t 0 -w 1 -b 127.0.0.1:5000 inference_server.server:app --access-logfile - --access-logformat '%(h)s %(t)s "%(r)s" %(s)s %(b)s'
