import argparse
import os
from functools import partial
from typing import List

import torch

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from transformers.generation_stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers.utils import is_offline_mode
from utils import GenerateRequest, GenerateResponse, TokenizeRequest, TokenizeResponse, print_rank_n, run_rank_n


class Model:
    def __init__(self, args: argparse.Namespace) -> None:
        self.tokenizer = None
        self.pad = None
        self.model = None
        self.input_device = None
        raise NotImplementedError("This is a dummy class")

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        input_tokens = self.tokenizer(request.text, return_tensors="pt", padding=True)

        for t in input_tokens:
            if torch.is_tensor(input_tokens[t]):
                input_tokens[t] = input_tokens[t].to(self.input_device)

        stopping_criteria = get_stopping_criteria(request.stop_sequences, self.tokenizer)

        with torch.no_grad():
            output = self.model.generate(
                **input_tokens,
                min_length=request.min_length,
                do_sample=request.do_sample,
                early_stopping=request.early_stopping,
                num_beams=request.num_beams,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                typical_p=request.typical_p,
                repetition_penalty=request.repetition_penalty,
                bos_token_id=request.bos_token_id,
                pad_token_id=request.pad_token_id,
                eos_token_id=request.eos_token_id,
                length_penalty=request.length_penalty,
                no_repeat_ngram_size=request.no_repeat_ngram_size,
                encoder_no_repeat_ngram_size=request.encoder_no_repeat_ngram_size,
                num_return_sequences=request.num_return_sequences,
                max_time=request.max_time,
                max_new_tokens=request.max_new_tokens,
                decoder_start_token_id=request.decoder_start_token_id,
                num_beam_groups=request.num_beam_groups,
                diversity_penalty=request.diversity_penalty,
                forced_bos_token_id=request.forced_bos_token_id,
                forced_eos_token_id=request.forced_eos_token_id,
                exponential_decay_length_penalty=request.exponential_decay_length_penalty,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
            )

        output_tokens = output.sequences

        input_token_lengths = [x.shape[0] for x in input_tokens.input_ids]

        check_max_input_length(input_token_lengths, request.max_input_length)

        output_token_lengths = [x.shape[0] for x in output_tokens]
        generated_tokens = [o - i for i, o in zip(input_token_lengths, output_token_lengths)]

        if request.remove_input_from_output:
            # the generate method's output includes input too. Remove input if
            # that is requested by the user
            output_tokens = [x[-i:] for x, i in zip(output_tokens, generated_tokens)]

        output_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

        return GenerateResponse(text=output_text, num_generated_tokens=generated_tokens)

    def tokenize(self, request: TokenizeRequest) -> TokenizeResponse:
        output = self.tokenizer(request.text, padding=request.padding)
        return TokenizeResponse(token_ids=output.input_ids, attention_mask=output.attention_mask)

    def shutdown(self) -> None:
        print_rank_n("shutting down")
        exit()


def get_downloaded_model_path(model_name: str):
    f = partial(
        snapshot_download,
        repo_id=model_name,
        allow_patterns=["*"],
        local_files_only=is_offline_mode(),
        cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
    )
    # download only on 1 process
    run_rank_n(f, barrier=True)
    # now since the snapshot is downloaded, pass the model_path to all processes
    return f()


def check_max_input_length(input_token_lengths: List[int], max_input_length: int) -> None:
    if max_input_length is None:
        return

    for i in input_token_lengths:
        if i > max_input_length:
            raise Exception(f"max supported input length = {max_input_length} for now")


class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, stop_sequences: List[List[int]]):
        self.stop_sequences = stop_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.shape[0] != 1:
            raise NotImplementedError("stop_sequences only support batch_size = 1")

        input_ids = input_ids[0].tolist()
        for s in self.stop_sequences:
            if input_ids[-len(s) :] == s:
                return True
        return False


def get_stopping_criteria(stop_sequences: List[str], tokenizer: AutoTokenizer) -> StoppingCriteriaList:
    stopping_criteria = StoppingCriteriaList()
    if stop_sequences is not None:
        stopping_criteria.append(StopSequenceCriteria(tokenizer(stop_sequences).input_ids))

    return stopping_criteria
