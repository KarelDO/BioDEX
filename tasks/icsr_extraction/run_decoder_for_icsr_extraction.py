import torch
import logging
import os
import copy
from typing import Optional
from dataclasses import dataclass, field
from transformers import (
    CONFIG_MAPPING,
    MODEL_WITH_LM_HEAD_MAPPING,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizer,
    TextDataset,
    Trainer,
    TrainingArguments,
    set_seed,
    GPT2LMHeadModel,
    AutoModelForCausalLM,
    # DataCollatorWithPadding
    DataCollatorForSeq2Seq,
)
import numpy as np
from transformers.trainer_utils import get_last_checkpoint
from datasets import load_dataset, load_from_disk
from tqdm import tqdm


# from sum_data_collator import DataCollatorForSumLanguageModeling
# from sum_dataset import LineByLineSumTextDataset

logger = logging.getLogger(__name__)


@dataclass
class PredictionArguments:
    """
    Arguments for model prediction
    """

    per_device_predict_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size to use when predicting."},
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams to use."},
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Repetition penalty."},
    )


@dataclass
class ModelArguments:
    """
    Arguments for the model
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Leave None if you want to train a model from"
                " scratch."
            )
        },
    )

    tokenizer_name: Optional[str] = field(
        default="gpt2",
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataArguments:
    """
    Arguments for data
    """

    # train_data_file: Optional[str] = field(
    #     default=None, metadata={"help": "The input training data file (a text file)."}
    # )
    # eval_data_file: Optional[str] = field(
    #     default=None,
    #     metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    # )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."
        },
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."
        },
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )
    max_target_length: Optional[int] = field(
        default=510, metadata={"help": "the max target length for summarization data. "}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization."
                "The training dataset will be truncated in block of this size for training."
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


def finetune():
    # parse args
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, PredictionArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        prediction_args,
        training_args,
    ) = parser.parse_args_into_dataclasses()

    os.environ['WANDB_PROJECT'] = 'biodex'

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # set seed
    set_seed(training_args.seed)
    # set up model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    print(config)
    # config.reorder_and_upcast_attn = True
    # config.scale_attn_by_inverse_layer_idx = True
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    # initial_weights = f"{model_args.model_name_or_path}/pytorch_model.bin"
    # model.load_state_dict(torch.load(initial_weights, map_location=torch.device("cpu")))
    # set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    # add extra pad token
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.add_special_tokens({"bos_token": "<|startoftext|>"})
    tokenizer.add_special_tokens({"sep_token": "[SEP]"})
    separator = tokenizer(tokenizer.sep_token, add_special_tokens=False)["input_ids"][0]
    eos_idx = tokenizer(tokenizer.eos_token, add_special_tokens=False)["input_ids"][0]
    # for x in range(1,10):
    # tokenizer.add_token(f"<|prefix{x}|>")
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    # set up data collator
    # data_collator = DataCollatorForSumLanguageModeling(tokenizer=tokenizer)
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length= data_args.max_source_length + data_args.max_target_length + 2)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        max_length=data_args.max_source_length + data_args.max_target_length + 2,
    )
    # set up data sets
    # train_dataset = get_dataset(data_args, tokenizer=tokenizer, training_args=training_args)
    # eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)

    if os.path.isdir(data_args.dataset_name):
        logger.info(f"Loading dataset from disk: {data_args.dataset_name}.")
        raw_datasets = load_from_disk(
            data_args.dataset_name,
        )
    else:
        logger.info(f"Loading dataset from Hugging Face Hub: {data_args.dataset_name}.")
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=None,
        )

    def preprocess_function(examples):
        # tokenize input text and summary
        text_encoding = tokenizer(
            examples[data_args.text_column],
            max_length=data_args.max_source_length,
            padding=False,
            truncation=True,
            is_split_into_words=False,
        )["input_ids"]
        summary_encoding = tokenizer(
            examples[data_args.summary_column],
            max_length=data_args.max_target_length,
            padding=False,
            truncation=True,
            is_split_into_words=False,
        )["input_ids"]
        # concatenate
        edited_sents = []
        for t, s in zip(text_encoding, summary_encoding):
            sent = t + [separator] + s + [eos_idx]
            edited_sents.append(sent)

        # mask labels as to not train on the input text
        labels = copy.deepcopy(edited_sents)
        for i, l in enumerate(labels):
            sep_idx = l.index(separator) + 1
            labels[i][:sep_idx] = [-100] * sep_idx

        return {"input_ids": edited_sents, "labels": labels}

    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="eval dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on eval dataset",
            )

    if training_args.do_predict:
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples
            )
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    # set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # launch fine tuning
    if training_args.do_train:

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        max_eval_samples = (
            data_args.max_eval_samples
            if data_args.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # assume batch size 1 to avoid issues with padding the source sentences in ways unseen during training
        batch_size = 1

        generated_texts = []
        sentences = predict_dataset[data_args.text_column]

        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sentences = sentences[i : i + batch_size]

            # encode
            input_ids = tokenizer(
                batch_sentences,
                return_tensors="pt",
                truncation=True,
                padding=True if batch_size > 1 else False,
                max_length=data_args.max_source_length,
            )["input_ids"]
            input_ids = input_ids.to(model.device)

            # add SEP token, just like we did in training
            batch_sep = (
                torch.tensor([separator] * len(input_ids))
                .unsqueeze(-1)
                .to(input_ids.device)
            )
            input_ids = torch.cat([input_ids, batch_sep], dim=1)

            # Set the forced beginning of sentence (BOS) token IDs
            # forced_bos_token_id = tokenizer.convert_tokens_to_ids(["I", "reactions:"])

            # generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_length=data_args.max_source_length
                    + 2
                    + data_args.max_target_length,
                    num_return_sequences=1,
                    # forced_bos_token_id=forced_bos_token_id,
                    do_sample=True,
                    num_beams=prediction_args.num_beams,  # Set the number of beams for beam search
                    repetition_penalty=prediction_args.repetition_penalty,
                    early_stopping=True,  # Enable early stopping of generation
                    pad_token_id=tokenizer.pad_token_id,
                )

            # extract the generated part
            outputs = [out[input_ids.shape[1] :] for out in outputs]
            # Convert generated output back into text
            batch_generated_texts = tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            generated_texts.extend(batch_generated_texts)

        # save outputs
        output_prediction_file = os.path.join(
            training_args.output_dir, "generated_predictions.txt"
        )
        generated_texts = [text.replace("\n", " ").strip() for text in generated_texts]
        with open(output_prediction_file, "w") as writer:
            writer.write("\n".join(generated_texts))

    # if training_args.do_predict:
    #     logger.info("*** Predict ***")

    #     # predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", **gen_kwargs)
    #     predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")
    #     metrics = predict_results.metrics
    #     max_predict_samples = (
    #         data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
    #     )
    #     metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

    #     trainer.log_metrics("predict", metrics)
    #     trainer.save_metrics("predict", metrics)

    #     if trainer.is_world_process_zero():
    #         # NOTE: need to change this
    #         if training_args.predict_with_generate:
    #             predictions = predict_results.predictions
    #             predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    #             predictions = tokenizer.batch_decode(
    #                 predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
    #             )
    #             predictions = [pred.strip() for pred in predictions]
    #             output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
    #             with open(output_prediction_file, "w") as writer:
    #                 writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    # if data_args.lang is not None:
    #     kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    finetune()
