import copy
import gc
import logging
from dataclasses import dataclass
from typing import Callable

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm
from transformers import (
    DynamicCache,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    set_seed,
)

from nanoflrt.utils import (
    AttackBuffer,
    find_executable_batch_size,
)

logger = logging.getLogger("nanoflrt")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class FLRTConfig:
    k1: int = 16
    k2: int = 64
    num_steps: int = 500
    init_len: int = 10
    buffer_size: int = 10
    p_add: float = 0.5
    p_swap: float = 0.25
    p_del: float = 0.25
    seed: int | None = 42
    verbose: bool = False
    use_kv_cache: bool = True
    activation_loss_weight: float = 1.0


@dataclass
class FLRTResult:
    best_string: str
    best_loss: float
    losses: list[float]
    strings: list[str]


class FLRT:
    """Implementation of the FLRT optimizer from https://arxiv.org/abs/2407.17447 with the slight
    change that we replace the worst candidate from the buffer, and don't use the additional
    loss functions outlined in the FLRT paper.

    NOTE: This optimizer is designed to be run with a single batch
        and num steps (in optimizer config) to be set to large values.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        config: FLRTConfig,
        activation_loss_fn: Callable | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.activation_loss_fn = activation_loss_fn

    def run(
        self,
        messages: str | list[dict[str, str]],
        target: str,
    ) -> FLRTResult:
        """Main step function for the FLRT optimizer.

        Note that this optimizer is designed to be run with a single batch and num steps (in
        optimizer config) to be set to large values. Each call to step starts the optimization from
        scratch.
        """
        if self.config.seed is not None:
            set_seed(self.config.seed)
            torch.use_deterministic_algorithms(True, warn_only=True)

        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        else:
            messages = copy.deepcopy(messages)

        # Append the GCG string at the end of the prompt if location not specified
        if not any(["{optim_str}" in d["content"] for d in messages]):
            messages[-1]["content"] = messages[-1]["content"] + "{optim_str}"

        template = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # Remove the BOS token -- this will get added when tokenizing, if necessary
        if self.tokenizer.bos_token and template.startswith(self.tokenizer.bos_token):
            template = template.replace(self.tokenizer.bos_token, "")
        before_str, after_str = template.split("{optim_str}")

        before_ids = self.tokenizer([before_str], padding=False, return_tensors="pt")[
            "input_ids"
        ].to(self.model.device, torch.int64)
        after_ids = self.tokenizer(
            [after_str], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device, torch.int64)
        target_ids = self.tokenizer(
            [target], add_special_tokens=False, return_tensors="pt"
        )["input_ids"].to(self.model.device, torch.int64)

        # Embed everything that doesn't get optimized
        embedding_layer = self.model.get_input_embeddings()
        before_embeds, after_embeds, target_embeds = (
            embedding_layer(ids) for ids in (before_ids, after_ids, target_ids)
        )

        buffer = AttackBuffer(
            self.tokenizer,
            init_len=self.config.init_len,
            size=self.config.buffer_size,
            device=self.model.device,
        )

        # Populate the KV Cache for tokens that appear before the optimized tokens
        if self.config.use_kv_cache:
            with torch.no_grad():
                self.kv_cache = self.model(
                    inputs_embeds=before_embeds,
                    return_legacy_cache=True,
                    use_cache=True,
                ).past_key_values

        losses = []
        optim_strings = []
        optim_idss = []

        for i in tqdm(range(self.config.num_steps)):
            best_ids = buffer.get_best().squeeze(0)

            rand = torch.rand(1, device=self.model.device).item()
            if rand < self.config.p_add or best_ids.shape[0] < 5:
                op = "add"
            elif rand < self.config.p_add + self.config.p_swap:
                op = "swap"
            else:
                op = "delete"

            logger.debug(f"Applying op: {op}")

            candidate_idxs = torch.randint(0, best_ids.shape[0], (self.config.k1,))

            if op == "delete":
                new_attack_ids_list = []
                for idx in candidate_idxs:
                    new_ids = torch.cat(
                        (best_ids[:idx], best_ids[idx + 1 :]), dim=0
                    ).unsqueeze(0)
                    new_attack_ids_list.append(new_ids)
                new_attack_ids = torch.cat(new_attack_ids_list, dim=0)
            else:
                input_embeds = embedding_layer(best_ids.unsqueeze(0))
                candidate_ids = self.sample_candidates(
                    candidate_idxs,
                    self.config.k2,
                    input_embeds,
                    before_embeds,
                )
                if op == "swap":
                    new_attack_ids_list = []
                    logger.debug(
                        f"candidate_ids: {candidate_ids}, shape: {candidate_ids.shape}"
                    )
                    for idx in range(candidate_ids.shape[0]):
                        swap_idx = candidate_idxs[idx]
                        new_ids = best_ids.clone()
                        new_ids[swap_idx] = candidate_ids[idx]
                        new_attack_ids_list.append(new_ids.unsqueeze(0))
                    new_attack_ids = torch.cat(new_attack_ids_list, dim=0)
                elif op == "add":
                    new_attack_ids_list = []
                    for idx in range(candidate_ids.shape[0]):
                        add_idx = candidate_idxs[idx]
                        new_ids = torch.cat(
                            [
                                best_ids[: add_idx + 1],
                                candidate_ids[idx : idx + 1],
                                best_ids[add_idx + 1 :],
                            ],
                            dim=0,
                        ).unsqueeze(0)
                        new_attack_ids_list.append(new_ids)
                    new_attack_ids = torch.cat(new_attack_ids_list, dim=0)

            new_ids = self.fixed_point_ids(new_attack_ids)
            input_embeds = torch.cat(
                [
                    embedding_layer(new_ids),
                    after_embeds.repeat(new_ids.shape[0], 1, 1),
                    target_embeds.repeat(new_ids.shape[0], 1, 1),
                ],
                dim=1,
            )

            loss = find_executable_batch_size(
                self.compute_candidates_loss, new_ids.shape[0]
            )(
                input_embeds=input_embeds,
                target_ids=target_ids,
            )

            # Take the bottom k1 from loss and update the buffer. Ignore the old buffer losses
            sorted_indices = loss.argsort()
            optim_id = new_ids[sorted_indices[0]]
            optim_str = self.tokenizer.decode(optim_id)
            optim_loss = loss[sorted_indices[0]]

            buffer.replace_worst(optim_id, optim_loss)

            losses.append(optim_loss.item())
            optim_strings.append(optim_str)
            optim_idss.append(optim_id.tolist())

            logger.debug(f"Buffer min loss: {buffer.losses.min().item()}")
            opstr = optim_str.replace("\n", r"\n")
            logger.debug(f"Step {i}: Optim String = {opstr}")

        best_id: Float[Tensor, "n_optim_ids"] = buffer.get_best()
        best_params = (
            torch.nn.functional.one_hot(optim_id, embedding_layer.num_embeddings)
            .to(
                dtype=embedding_layer.weight.dtype, device=embedding_layer.weight.device
            )
            .unsqueeze(0)
        )
        assert best_params.shape[0] == 1
        assert best_params.shape[2] == embedding_layer.num_embeddings

        return FLRTResult(
            best_string=self.tokenizer.decode(best_id),
            best_loss=min(losses),
            losses=losses,
            strings=optim_strings,
        )

    def sample_candidates(
        self,
        candidate_idxs: Tensor,
        k2: int,
        input_embeds: Tensor,
        before_embeds: Tensor | None = None,
    ):
        with torch.no_grad():
            if self.config.use_kv_cache:
                outputs = self.model(
                    inputs_embeds=input_embeds,
                    past_key_values=DynamicCache.from_legacy_cache(self.kv_cache),
                    output_hidden_states=False,
                )
                logits = outputs.logits
            else:
                assert before_embeds is not None, (
                    "before_embeds must be provided if kv_cache is None"
                )
                input_embeds = torch.cat([before_embeds, input_embeds], dim=1)
                outputs = self.model(
                    inputs_embeds=input_embeds, output_hidden_states=False
                )
                logits = outputs.logits[..., before_embeds.shape[1] :, :]

        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze(0)
        special_ids = [0, 1, 2]  # Hardcoded from tokenizer.all_special_ids for now
        probs[..., special_ids] = 0.0
        probs[..., self.tokenizer.vocab_size :] = 0.0
        sampled_ids = torch.multinomial(
            probs[candidate_idxs], num_samples=k2, replacement=False
        )
        selection = torch.randint(0, k2, (candidate_idxs.shape[0],))
        return sampled_ids[torch.arange(candidate_idxs.shape[0]), selection]

    def filter_ids(self, ids: Tensor) -> torch.Tensor:
        """Filters out sequeneces of token ids that change after retokenization.

        Args:
            ids : Tensor, shape = (search_width, n_optim_ids)
                token ids

        Returns:
            filtered_ids : Tensor, shape = (new_search_width, n_optim_ids)
                all token ids that are the same after retokenization
        """
        ids_decoded = self.tokenizer.batch_decode(ids)
        filtered_ids = []

        for i in range(len(ids_decoded)):
            # Retokenize the decoded token ids
            ids_encoded = self.tokenizer(
                ids_decoded[i], return_tensors="pt", add_special_tokens=False
            ).to(ids.device)["input_ids"][0]
            if torch.equal(ids[i], ids_encoded):
                filtered_ids.append(ids[i])

        if len(filtered_ids) == 0:
            return []
        else:
            return torch.stack(filtered_ids)

    def fixed_point_ids(self, ids: Tensor) -> torch.Tensor:
        is_fixed_point = False
        ids_encoded = ids
        while not is_fixed_point:
            is_fixed_point = True
            ids_decoded = self.tokenizer.batch_decode(ids_encoded)
            encoded_ids = []

            for i in range(len(ids_decoded)):
                # Retokenize the decoded token ids
                ids_recoded = self.tokenizer(
                    ids_decoded[i], return_tensors="pt", add_special_tokens=False
                ).to(ids.device)["input_ids"][0]
                if not encoded_ids or len(encoded_ids[0]) == len(ids_recoded):
                    encoded_ids.append(ids_recoded)
                    if not torch.equal(ids_encoded[i], ids_recoded):
                        is_fixed_point = False

            ids_encoded = torch.stack(encoded_ids)

        return ids_encoded

    def compute_candidates_loss(
        self,
        search_batch_size: int,
        *,
        input_embeds: Tensor,
        target_ids: Tensor,
    ):
        all_loss = []
        prefix_cache_batch = []

        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]
                current_batch_size = input_embeds_batch.shape[0]

                if self.config.use_kv_cache and (
                    not prefix_cache_batch or current_batch_size != search_batch_size
                ):
                    prefix_cache_batch = [
                        [
                            x.expand(current_batch_size, -1, -1, -1)
                            for x in self.kv_cache[i]
                        ]
                        for i in range(len(self.kv_cache))
                    ]

                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=DynamicCache.from_legacy_cache(
                            prefix_cache_batch
                        ),
                        use_cache=True,
                        output_hidden_states=True,
                    )
                else:
                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        output_hidden_states=True,  # We need hidden states for activation loss
                    )

                logits = outputs.logits
                hidden_states = outputs.hidden_states

                tmp = input_embeds.shape[1] - target_ids.shape[1]
                shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
                shift_labels = target_ids.repeat(current_batch_size, 1)

                ce_loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                )
                ce_loss = ce_loss.view(current_batch_size, -1).mean(dim=-1)

                if self.activation_loss_fn is not None:
                    activation_loss = self.activation_loss_fn(hidden_states)
                    total_loss = (
                        ce_loss + self.config.activation_loss_weight * activation_loss
                    )
                else:
                    total_loss = ce_loss

                all_loss.append(total_loss)

                del outputs
                gc.collect()
                torch.cuda.empty_cache()

        return torch.cat(all_loss, dim=0)


def run(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    messages: str | list[dict[str, str]],
    target: str,
    config: FLRTConfig | None = None,
    activation_loss_fn: Callable | None = None,
) -> FLRTResult:
    """Generates a single optimized string using FLRT.

    Args:
        model: The model to use for optimization.
        tokenizer: The model's tokenizer.
        messages: The conversation to use for optimization.
        target: The target generation.
        config: The FLRT configuration to use.
        activation_loss_fn: Optional callable that takes hidden states and returns a scalar loss.
            The function should take a tuple(num_layers) of tensors(batch_size, sequence_length, hidden_size)
            and return a tensor of shape (batch_size,).

    Returns:
        A FLRTResult object that contains losses and the optimized strings.
    """
    if config is None:
        config = FLRTConfig()

    logger.setLevel(logging.DEBUG if config.verbose else logging.INFO)

    flrt = FLRT(model, tokenizer, config, activation_loss_fn)
    result = flrt.run(messages, target)
    return result
