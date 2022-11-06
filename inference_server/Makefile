gen-proto:
	pip install grpcio-tools==1.50.0

	mkdir -p model_handler/grpc_utils/pb

	python -m grpc_tools.protoc -Imodel_handler/grpc_utils/proto --python_out=model_handler/grpc_utils/pb --grpc_python_out=model_handler/grpc_utils/pb model_handler/grpc_utils/proto/generation.proto

	find model_handler/grpc_utils/pb/ -type f -name "*.py" -print0 -exec sed -i -e 's/^\(import.*pb2\)/from . \1/g' {} \;

	touch model_handler/grpc_utils/__init__.py
	touch model_handler/grpc_utils/pb/__init__.py

	rm -rf model_handler/grpc_utils/pb/*.py-e
