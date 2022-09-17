'''
Copyright 2022 The Microsoft DeepSpeed Team
'''
import asyncio
import time
import mii
from mii.utils import kwarg_dict_to_proto
from mii.grpc_related.proto import modelresponse_pb2


def mii_query_handle(deployment_name):
    """Get a query handle for a local deployment:

        mii/examples/local/gpt2-query-example.py
        mii/examples/local/roberta-qa-query-example.py


    Arguments:
        deployment_name: Name of the deployment. Used as an identifier for posting queries for ``LOCAL`` deployment.

    Returns:
        query_handle: A query handle with a single method `.query(request_dictionary)` using which queries can be sent to the model.

    """

    configs = mii.utils.import_score_file(deployment_name).configs

    task = configs[mii.constants.TASK_NAME_KEY]

    assert task is not None, "The task name should be set before calling init"

    return MIIServerClient(task,
                               "na",
                               "na",
                               mii_configs=configs[mii.constants.MII_CONFIGS_KEY],
                               initialize_service=False,
                               initialize_grpc_client=True,
                               use_grpc_server=True)


class MIIServerClient(mii.MIIServerClient):
    '''Initialize the model, setup the server and client for the model under model_path'''
    def __init__(self,
                 task_name,
                 model_name,
                 model_path,
                 ds_optimize=True,
                 ds_zero=False,
                 ds_config=None,
                 mii_configs={},
                 initialize_service=True,
                 initialize_grpc_client=True,
                 use_grpc_server=False):

        mii_configs = mii.config.MIIConfig(**mii_configs)

        self.task = mii.utils.get_task(task_name)

        self.num_gpus = self._get_num_gpus(mii_configs)
        assert self.num_gpus > 0, "GPU count must be greater than 0"

        # This is true in two cases
        # i) If its multi-GPU
        # ii) It is a local deployment
        self.use_grpc_server = True if (self.num_gpus > 1) else use_grpc_server
        self.initialize_service = initialize_service
        self.initialize_grpc_client = initialize_grpc_client

        self.port_number = mii_configs.port_number

        if initialize_service and not self.use_grpc_server:
            self.model = None

        if self.initialize_service:
            self.process = self._initialize_service(model_name,
                                                    model_path,
                                                    ds_optimize,
                                                    ds_zero,
                                                    ds_config,
                                                    mii_configs)
            if self.use_grpc_server:
                self._wait_until_server_is_live()

        if self.initialize_grpc_client and self.use_grpc_server:
            self.stubs = []
            self.asyncio_loop = asyncio.get_event_loop()
            self._initialize_grpc_client()

    #runs task in parallel and return the result from the first task
    async def _query_in_tensor_parallel(self, request_string, query_kwargs):
        responses = []
        for i in range(self.num_gpus):
            responses.append(
                self.asyncio_loop.create_task(
                    self._request_async_response(i,
                                                 request_string,
                                                 query_kwargs)))

        await responses[0]

        return responses[0]

    async def _request_async_response(self, stub_id, request_dict, query_kwargs):
        proto_kwargs = kwarg_dict_to_proto(query_kwargs)
        if self.task == mii.Tasks.TEXT_GENERATION:
            # convert to batch of queries if they are not already
            if not isinstance(request_dict['query'], list):
                request_dict['query'] = [request_dict['query']]
            req = modelresponse_pb2.MultiStringRequest(request=request_dict['query'],
                                                       query_kwargs=proto_kwargs)
            response = await self.stubs[stub_id].GeneratorReply(req)

        # elif self.task == mii.Tasks.TEXT_CLASSIFICATION:
        #     response = await self.stubs[stub_id].ClassificationReply(
        #         modelresponse_pb2.SingleStringRequest(request=request_dict['query'],
        #                                               query_kwargs=proto_kwargs))

        # elif self.task == mii.Tasks.QUESTION_ANSWERING:
        #     response = await self.stubs[stub_id].QuestionAndAnswerReply(
        #         modelresponse_pb2.QARequest(question=request_dict['question'],
        #                                     context=request_dict['context'],
        #                                     query_kwargs=proto_kwargs))
        # elif self.task == mii.Tasks.FILL_MASK:
        #     response = await self.stubs[stub_id].FillMaskReply(
        #         modelresponse_pb2.SingleStringRequest(request=request_dict['query'],
        #                                               query_kwargs=proto_kwargs))

        # elif self.task == mii.Tasks.TOKEN_CLASSIFICATION:
        #     response = await self.stubs[stub_id].TokenClassificationReply(
        #         modelresponse_pb2.SingleStringRequest(request=request_dict['query'],
        #                                               query_kwargs=proto_kwargs))

        # elif self.task == mii.Tasks.CONVERSATIONAL:
        #     response = await self.stubs[stub_id].ConversationalReply(
        #         modelresponse_pb2.ConversationRequest(
        #             text=request_dict['text'],
        #             conversation_id=request_dict['conversation_id']
        #             if 'conversation_id' in request_dict else None,
        #             past_user_inputs=request_dict['past_user_inputs'],
        #             generated_responses=request_dict['generated_responses'],
        #             query_kwargs=proto_kwargs))

        else:
            assert False, "unknown task"
        return response

    def _request_response(self, request_dict, query_kwargs):
        start = time.time()
        if self.task == mii.Tasks.TEXT_GENERATION:
            response = self.model(request_dict['query'], **query_kwargs)

        # elif self.task == mii.Tasks.TEXT_CLASSIFICATION:
        #     response = self.model(request_dict['query'], **query_kwargs)

        # elif self.task == mii.Tasks.QUESTION_ANSWERING:
        #     response = self.model(question=request_dict['query'],
        #                           context=request_dict['context'],
        #                           **query_kwargs)

        # elif self.task == mii.Tasks.FILL_MASK:
        #     response = self.model(request_dict['query'], **query_kwargs)

        # elif self.task == mii.Tasks.TOKEN_CLASSIFICATION:
        #     response = self.model(request_dict['query'], **query_kwargs)

        # elif self.task == mii.Tasks.CONVERSATIONAL:
        #     response = self.model(["", request_dict['query']], **query_kwargs)

        else:
            raise NotImplementedError(f"task is not supported: {self.task}")
        end = time.time()
        return f"{response}" + f"\n Model Execution Time: {end-start} seconds"

    async def query(self, request_dict, **query_kwargs):
        """Query a local deployment:

            mii/examples/local/gpt2-query-example.py
            mii/examples/local/roberta-qa-query-example.py

        Arguments:
            request_dict: A task specific request dictionary consistinging of the inputs to the models
            query_kwargs: additional query parameters for the model

        Returns:
            response: Response of the model
        """
        if not self.use_grpc_server:
            response = self._request_response(request_dict, query_kwargs)
            ret = f"{response}"
        else:
            assert self.initialize_grpc_client, "grpc client has not been setup when this model was created"
            response = self.asyncio_loop.run_until_complete(
                self._query_in_tensor_parallel(request_dict,
                                               query_kwargs))
            ret = response.result()
        return ret
