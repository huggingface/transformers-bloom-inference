import argparse
import copy
from typing import List, Union

import torch

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig

from ..utils import (
    ForwardRequest,
    ForwardResponse,
    GenerateRequest,
    GenerateResponse,
    TokenizeRequest,
    TokenizeResponse,
)


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model = None
        self.input_device = None
        self.max_input_length = args.max_input_length
        self.max_batch_size = args.max_batch_size

    def post_init(self, model_name: str) -> None:
        self.is_encoder_decoder = AutoConfig.from_pretrained(model_name).is_encoder_decoder
        self.generation_config = GenerationConfig.from_model_config(AutoConfig.from_pretrained(model_name))
        self.tokenizer = load_tokenizer(model_name)
        self.pad = self.tokenizer.pad_token_id
        self.prefix_token_id = self.tokenizer("A")["input_ids"][0]

    def get_generation_config(self, request: GenerateRequest) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config)
        request = dict(request)

        request_filtered = {}
        for key, value in request.items():
            if value is not None and key not in ["text", "remove_input_from_output"]:
                request_filtered[key] = value
        request_filtered["return_dict_in_generate"] = True

        generation_config.update(**request_filtered)
        return generation_config

    def generate(self, request: GenerateRequest) -> Union[GenerateResponse, Exception]:
        try:
            batch_size = len(request.text)

            check_batch_size(batch_size, self.max_batch_size)

            input_tokens = self.tokenizer(request.text, return_tensors="pt", padding=True)
            max_input_length_in_batch = input_tokens.input_ids[0].shape[0]

            check_max_input_length(max_input_length_in_batch, self.max_input_length)

            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(self.input_device)

            num_input_tokens = input_tokens["input_ids"].shape[1]

            generation_config = self.get_generation_config(request)

            output = self.model.generate(**input_tokens, generation_config=generation_config)

            output_tokens = output.sequences

            if self.is_encoder_decoder:
                num_generated_tokens = (output_tokens != self.pad).sum(dim=-1).tolist()
                generated_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
            else:
                generated_tokens = output_tokens[:, num_input_tokens:]
                num_generated_tokens = (generated_tokens != self.pad).sum(dim=-1).tolist()

                if request.remove_input_from_output:
                    # create the dummy prefix for detokenization
                    prefix_to_add = torch.tensor([[self.prefix_token_id]] * batch_size).to(self.input_device)
                    # the generate method's output includes input too. Remove input if
                    # that is requested by the user
                    generated_tokens = torch.cat([prefix_to_add, generated_tokens], dim=1)
                    generated_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    generated_text = [i[1:] for i in generated_text]
                else:
                    generated_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

            return GenerateResponse(
                text=generated_text,
                num_generated_tokens=num_generated_tokens,
                is_encoder_decoder=self.is_encoder_decoder,
            )
        except Exception as exception:
            return exception

    def forward(self, request: ForwardRequest) -> Union[ForwardResponse, Exception]:
        def prepare_tensors(conditioning_tokens: List[List[int]], response_tokens: List[List[int]]):
            bs = len(conditioning_tokens)

            input_ids = [conditioning_tokens[i] + response_tokens[i] for i in range(bs)]
            attention_mask = [[1] * (len(conditioning_tokens[i]) + len(response_tokens[i])) for i in range(bs)]
            labels = [[-100] * len(conditioning_tokens[i]) + response_tokens[i] for i in range(bs)]

            input_ids = pad(input_ids, self.tokenizer.pad_token_id)
            attention_mask = pad(attention_mask, 0)
            labels = pad(labels, -100)

            return {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels),
            }

        def pad(arrays: list, padding: int, max_length: int = None):
            if max_length is None:
                max_length = max(list(map(len, arrays)))

            arrays = [[padding] * (max_length - len(array)) + array for array in arrays]
            return arrays

        try:
            batch_size = len(request.conditioning_text)

            check_batch_size(batch_size, self.max_batch_size)

            conditioning_tokens = self.tokenizer(request.conditioning_text)["input_ids"]
            response_tokens = self.tokenizer(request.response)["input_ids"]

            max_length_in_batch = max([len(conditioning_tokens) + len(response_tokens)])
            check_max_input_length(max_length_in_batch, self.max_input_length)

            input_tokens = prepare_tensors(conditioning_tokens, response_tokens)

            for t in input_tokens:
                if torch.is_tensor(input_tokens[t]):
                    input_tokens[t] = input_tokens[t].to(self.input_device)

            loss = self.model(**input_tokens).loss

            return ForwardResponse(nll=loss.item(), is_encoder_decoder=self.is_encoder_decoder)
        except Exception as exception:
            return exception

    def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        return TokenizeResponse(
            token_ids=self.tokenizer(request.text).input_ids,
            is_encoder_decoder=self.is_encoder_decoder,
        )


def check_max_input_length(input_token_length: int, max_input_length: int) -> None:
    if max_input_length is None:
        return

    if input_token_length > max_input_length:
        raise Exception(f"max supported input length = {max_input_length} for now")


def check_batch_size(batch_size: int, max_batch_size: int) -> None:
    if max_batch_size is None:
        return

    if batch_size > max_batch_size:
        raise Exception(f"max supported batch size = {max_batch_size} for now")


# this is a hack for now
def get_hf_model_class(model_class: str) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
    return getattr(transformers, model_class)


def load_tokenizer(model_name: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    return tokenizer
