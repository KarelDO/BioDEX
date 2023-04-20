import torch
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
)
from datasets import load_dataset


from sum_data_collator import DataCollatorForSumLanguageModeling
from sum_dataset import LineByLineSumTextDataset


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
        default="gpt2", metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
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
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )
    max_target_length: Optional[int] = field(
        default=510, metadata={"help": "the max target length for summarization data. "}
    )
    # train_max_target_length: Optional[int] = field(
    #     default=510, metadata={"help": "the max target length for training data. "}
    # )
    # eval_max_target_length: Optional[int] = field(
    #     default=510, metadata={"help": "the max target length for dev data. "}
    # )
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


def get_dataset(
    args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    evaluate: bool = False,
    cache_dir: Optional[str] = None,
    training_args: TrainingArguments = None,
):
    # file_path = args.eval_data_file if evaluate else args.train_data_file
    dataset_name = args.dataset_name
    dataset_config_name = args.dataset_config_name
    max_source_length = args.max_source_length
    max_target_length = args.train_max_target_length if not evaluate else args.eval_max_target_length
    text_column = args.text_column
    summary_column = args.summary_column

    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        dataset_name,
        dataset_config_name,
        cache_dir=cache_dir,
    )
    if evaluate:
        split = raw_datasets['validation']
    else:
        split = raw_datasets['train']
    
    dataset = LineByLineSumTextDataset(
        tokenizer=tokenizer,
        split=split,
        text_column=text_column, 
        summary_column=summary_column, 
        block_size=512,
        bos_tok=tokenizer.bos_token,
        eos_tok=tokenizer.sep_token,
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        use_stream_mode=False
    )

    return dataset


def finetune():
    # parse args
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # set seed
    set_seed(training_args.seed)
    # set up model
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    print(config)
    #config.reorder_and_upcast_attn = True
    #config.scale_attn_by_inverse_layer_idx = True
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
    separator = tokenizer(tokenizer.sep_token, add_special_tokens=False)['input_ids'][0]
    eos_idx = tokenizer(tokenizer.eos_token, add_special_tokens=False)['input_ids'][0]
    #for x in range(1,10):
        #tokenizer.add_token(f"<|prefix{x}|>")
    embedding_layer = model.resize_token_embeddings(len(tokenizer))
    # set up data collator
    data_collator = DataCollatorForSumLanguageModeling(tokenizer=tokenizer)
    # set up data sets
    # train_dataset = get_dataset(data_args, tokenizer=tokenizer, training_args=training_args)
    # eval_dataset = get_dataset(data_args, tokenizer=tokenizer, evaluate=True)

    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=None,
    )

    padding = False
    def preprocess_function(examples):
        # tokenize input text and summary
        text_encoding = tokenizer(examples['abstract'], max_length=data_args.max_source_length, padding=padding, truncation=True, is_split_into_words=False)['input_ids']
        summary_encoding = tokenizer(examples['target'], max_length=data_args.max_target_length, padding=padding, truncation=True, is_split_into_words=False)['input_ids']
        # concatenate
        edited_sents = []
        for t, s in zip(text_encoding, summary_encoding):
            sent = t + [separator] + s + [eos_idx]
            edited_sents.append(sent)

        # mask labels
        labels = copy.deepcopy(edited_sents)
        for i,l in enumerate(labels):
            sep_idx = l.index(separator) + 1
            labels[i][:sep_idx] = [-100] * sep_idx

        return {
            "input_ids": edited_sents,
            "labels": labels
        }
    

    train_dataset = raw_datasets['train'].map(preprocess_function, batched=True, remove_columns=raw_datasets['train'].column_names)
    eval_dataset = raw_datasets['validation'].map(preprocess_function, batched=True, remove_columns=raw_datasets['validation'].column_names)
    
    # set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        # data_collator=data_collator
    )
    # launch fine tuning
    #trainer.train(resume_from_checkpoint=f"{model_args.model_name_or_path}")
    trainer.train()
    # save final model
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    finetune()
