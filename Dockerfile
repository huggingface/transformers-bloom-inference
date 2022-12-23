FROM nvidia/cuda:11.6.0-devel-ubi8 as cuda

ENV PORT=5000

WORKDIR /src

FROM cuda as conda

# taken form pytorch's dockerfile
RUN curl -L -o ./miniconda.sh -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ./miniconda.sh && \
    ./miniconda.sh -b -p /opt/conda && \
    rm ./miniconda.sh

ENV PYTHON_VERSION=3.9 \
    PATH=/opt/conda/envs/inference/bin:/opt/conda/bin:${PATH}

# create conda env
RUN conda create -n inference python=${PYTHON_VERSION} pip -y

# change shell to activate env
SHELL ["conda", "run", "-n", "inference", "/bin/bash", "-c"]

FROM conda as conda_env

# update conda
RUN conda update -n base -c defaults conda -y

# necessary stuff
RUN pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116 \
    transformers \
    deepspeed==0.7.5 \
    deepspeed-mii==0.0.2 \
    accelerate \
    gunicorn \
    flask \
    flask_api \ 
    pydantic \
    huggingface_hub \
	grpcio-tools==1.50.0 \
    --no-cache-dir

# copy the code
COPY inference_server inference_server
COPY Makefile Makefile
COPY LICENSE LICENSE

# install grpc and compile protos
RUN make gen-proto

# clean conda env
RUN conda clean -ya

EXPOSE ${PORT}

# change this as you like 🤗
ENV TRANSFORMERS_CACHE=/transformers_cache/ \
    HUGGINGFACE_HUB_CACHE=${TRANSFORMERS_CACHE} \
    HOME=/homedir

RUN mkdir ${HOME} && chmod g+wx ${HOME} && \
    mkdir tmp && chmod -R g+w tmp

# for debugging
# RUN chmod -R g+w inference_server && chmod g+w Makefile

CMD make bloom-176b
