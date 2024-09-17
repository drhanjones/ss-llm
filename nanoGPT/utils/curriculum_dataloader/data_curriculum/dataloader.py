import string
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer
from .pacing_fn import get_pacing_fn
from typing import Sequence, Any
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, DatasetDict
import tqdm


POS_TAG_MAP = {
    "NOUN": 0,
    "VERB": 1,
    "ADJ": 2,
    "ADV": 3,
    "PRON": 4,
    "DET": 5,
    "ADP": 6,
    "NUM": 7,
    "CONJ": 8,
    "PRT": 9,
    ".": 10,
    "X": 11,
}

SPOKEN_FIRST_DATASET_ORDER = {
    "aochildes.txt": 1,
    "bnc_spoken.txt": 2,
    "switchboard.txt": 2,
    "open_subtitles.txt": 3,
    "qed.txt": 3,
    "cbt.txt": 4,
    "children_stories.txt": 4,
    "simple_wikipedia.txt": 5,
    "wikipedia.txt": 6,
    "gutenberg.txt": 6,
}

GRAMMATICAL_FIRST_DATASET_ORDER = {
    "cbt.txt": 1,
    "children_stories.txt": 1,
    "simple_wikipedia.txt": 2,
    "wikipedia.txt": 3,
    "gutenberg.txt": 3,
    "open_subtitles.txt": 4,
    "bnc_spoken.txt": 5,
    "switchboard.txt": 5,
    "qed.txt": 6,
    "aochildes.txt": 6,
}

class DatasetPreprocessor(object):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 max_input_length: int,
                 include_punctuation: bool = True,
                 join_sentences: bool = True,
                 dataset_subconfig: str = "original_strict_small",
                 spoken_first: bool = True,):

        """
        Args:
            tokenizer (PreTrainedTokenizer): instantiated tokenizer object
            include_punctuation (bool): whether to include punctuation in the dataset
            max_input_length (int): maximum input length for the model
            join_sentences (bool): whether to join sentences together
            dataset_subconfig (str): subconfig of the dataset
            spoken_first (bool): whether to put spoken datasets first ordering or grammatical datasets first ordering
            
            -One of 'original_strict_small', 'strict_small', 'sem_strict_small', 'original_strict_small_gold', 'strict_small_gold',
             Default uses 'strict_small', I think we should use "original_strict_small" so that we ignore all the pos tags and just use the text
             Also there's difference in both, where original doesn't have multiple lines concatenated, as mentioned in paper (5 lines for speech datasets)

            Returns:
            -A dictionary containing the following keys with batches of data as values:
                * input_ids: list of input ids
                * output_ids: list of output ids
                * special_tokens_mask: list of special tokens mask
                * attention_mask: list of attention mask
                * pos_tags: list of pos tags
                * filename: list of filenames
                * curriculum_order: list of curriculum orders
        """

        # data processing params
        self.include_punctuation = include_punctuation
        self.max_input_length = max_input_length
        self.join_sentences = join_sentences
        if not self.join_sentences:
            raise NotImplementedError(
                "Currently, only join_sentences=True is supported for CLM tasks."
            )
        self.dataset_subconfig = dataset_subconfig
        self.curriculum_dataset_order = (
            SPOKEN_FIRST_DATASET_ORDER
            if spoken_first
            else GRAMMATICAL_FIRST_DATASET_ORDER
        )

        self.tokenizer = tokenizer

    ### --- Callback functions --- ###

    # NOTE: The function names of callbacks must match the names in the data preprocessing
    # callback_functions list (sepcified in the config file)

    ### --- Callback functions --- ###

    def __call__(self, examples):
        if not self.include_punctuation:
            examples["text"] = [
                line.translate(str.maketrans("", "", string.punctuation))
                for line in examples["text"]
            ]

        batch = {
            "input_ids": [],
            "output_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "pos_tags": [],
            "filename": [],
            "curriculum_order": []
        }

        full_tokenized_inputs = {
            "input_ids": [],
            "output_ids": [],
            "special_tokens_mask": [],
            "attention_mask": [],
            "pos_tags": [],
            "filename": [],
            "curriculum_order": []
        }

        for example in range(len(examples["text"])):
            text = examples["text"][example]
            tagged_text = examples["tagged_text"][example]
            filename = examples["filename"][example]

            tokenized_inputs = self.tokenizer(
                text,
                pad_to_multiple_of=self.max_input_length
                if not self.join_sentences
                else None,
                padding="longest" if not self.join_sentences else "do_not_pad",
                max_length=self.max_input_length
                if not self.join_sentences
                else None,
                truncation=False,
                return_special_tokens_mask=True,
                return_offsets_mapping=True,
            )

            # Original dataset doesn't have pos tags
            if "original" in self.dataset_subconfig:
                pos_tags = [POS_TAG_MAP["X"]] * len(
                    tokenized_inputs["input_ids"]
                )
            else:
                subwords = [text[offset[0] : offset[1]] for offset in tokenized_inputs["offset_mapping"]]  # type: ignore
                tag_pairs = [
                    tag_pair.split("__<label>__")
                    for tag_pair in tagged_text.strip().split(" ")
                    if tag_pair != ""
                ]
                # Iterate through subwords and assign POS tags, hopefully they should match up, since
                # the subwords in example_tagged_text were extracted by the tokenizer in the first place
                pos_tags = []
                i = 0
                for subword in subwords:
                    # This indicates that the subword is a special token
                    if subword == "" or subword == "\n":
                        pos_tags.append(POS_TAG_MAP["X"])
                        continue
                    # Check if we're at the start of the next word
                    if i + 1 < len(tag_pairs) and tag_pairs[i + 1][
                        0
                    ].startswith(subword):
                        i += 1
                    # Keep using the POS tag of the current word
                    pos_tags.append(
                        POS_TAG_MAP[tag_pairs[i][1]]
                        if tag_pairs[i][1] in POS_TAG_MAP
                        else POS_TAG_MAP["X"]
                    )

            if self.join_sentences:
                full_tokenized_inputs["input_ids"].extend(
                    tokenized_inputs["input_ids"]
                )
                full_tokenized_inputs["special_tokens_mask"].extend(
                    tokenized_inputs["special_tokens_mask"]
                )
                full_tokenized_inputs["attention_mask"].extend(
                    tokenized_inputs["attention_mask"]
                )
                full_tokenized_inputs["pos_tags"].extend(pos_tags)
                full_tokenized_inputs["filename"].extend(
                    [filename] * len(tokenized_inputs["input_ids"])
                )
                full_tokenized_inputs["curriculum_order"].extend(
                    [self.curriculum_dataset_order[filename]] * len(tokenized_inputs["input_ids"])
                )

            else:
                # Split into multiple examples if the input is too long
                for i in range(
                    0,
                    len(tokenized_inputs["input_ids"]),
                    self.max_input_length,
                ):
                    # Check if the final example would contain only special tokens and if so, don't include it
                    if (
                        sum(
                            tokenized_inputs["special_tokens_mask"][
                                i : i + self.max_input_length
                            ]
                        )
                        == self.max_input_length
                    ):
                        break
                    batch["input_ids"].append(
                        tokenized_inputs["input_ids"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["special_tokens_mask"].append(
                        tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["attention_mask"].append(
                        tokenized_inputs["attention_mask"][i : i + self.max_input_length]  # type: ignore
                    )
                    batch["pos_tags"].append(
                        pos_tags[i : i + self.max_input_length]
                    )
                    batch["filename"].append(filename)
                    batch["curriculum_order"].append(self.curriculum_dataset_order[filename])
                # Need to do extra padding for pos tags because the tokenizer padding doesn't work on them
                if len(batch["pos_tags"][-1]) < self.max_input_length:
                    batch["pos_tags"][-1].extend(
                        [POS_TAG_MAP["X"]]
                        * (self.max_input_length - len(batch["pos_tags"][-1]))
                    )

        if self.join_sentences:
            # NOTE: We drop the last batch if it's not full. This is just to ensure every example is the same length which makes things easier.
            truncated_length = (
                len(full_tokenized_inputs["input_ids"])
                // self.max_input_length
            ) * self.max_input_length

            for i in range(0, truncated_length, self.max_input_length):

                if i + 1 + self.max_input_length > len(full_tokenized_inputs["input_ids"]):
                    #Just to ensure output_ids is not out of bounds
                    break
                batch["input_ids"].append(
                    full_tokenized_inputs["input_ids"][i : i + self.max_input_length]  # type: ignore
                )
                batch["output_ids"].append(
                    full_tokenized_inputs["input_ids"][i + 1 : i + self.max_input_length + 1]  # type: ignore
                )
                batch["special_tokens_mask"].append(
                    full_tokenized_inputs["special_tokens_mask"][i : i + self.max_input_length]  # type: ignore
                )
                batch["attention_mask"].append(
                    full_tokenized_inputs["attention_mask"][i : i + self.max_input_length]  # type: ignore
                )
                batch["pos_tags"].append(
                    full_tokenized_inputs["pos_tags"][i : i + self.max_input_length]  # type: ignore
                )
                batch["filename"].append(full_tokenized_inputs["filename"][i])
                batch["curriculum_order"].append(full_tokenized_inputs["curriculum_order"][i])

        #COMMENTED OUT CALLBACK FUNCTIONS FOR NOW
        # if self.callback_functions:
        #     for callback_function in self.callback_functions:
        #         examples[callback_function] = getattr(self, callback_function)(
        #             examples
        #         )

        return batch



def load_cc_dataset(dataset_name,
                    dataset_subconfig: str = 'original_strict_small',
                    load_from_hf: bool = True,) -> DatasetDict:
    """
    Loads the dataset from Hugging Face or from a local file. If loading from Hugging Face, ensure HF_READ_TOKEN is set in the environment.

    Args:
        * load_from_hf (bool): whether to load the dataset from Hugging Face
        * dataset_name (str): the name of the dataset to load
        * dataset_subconfig (str): the subconfig of the dataset to load
    Returns:
        * dataset (DatasetDict): the loaded dataset
    """
    dataset = None
    if load_from_hf:
        HF_READ_TOKEN = "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" #Don't hardcode
        #dataset_name = 'cambridge-climb/BabyLM'
        #dataset_subconfig = 'original_strict_small'  # actual code uses 'strict_small' sticking to "original_strict_small" for now
        try:
            dataset: DatasetDict = load_dataset(
                dataset_name,
                dataset_subconfig,
                token=HF_READ_TOKEN,
            )  # type: ignore

        except Exception as e:
            print(f"Error loading dataset:, switching to local file. Error: {e}")

    return dataset

def get_curriculum_end_indices(dataset):
    """
    Returns the end indices of each level of the curriculum in the dataset.
    :param dataset:
    :return:
        curriculum_end_indices (Dict[str, int]): the end indices of each curriculum in the dataset
        Length of the curriculum_end_indices should be equal to the number of curriculums in the dataset
    """

    curriculum_end_indices = {}
    for i, order in tqdm.tqdm(enumerate(dataset["curriculum_order"])):
        curriculum_end_indices[order] = i + 1

    #Sanity check
    assert len(curriculum_end_indices) == len(set(dataset["curriculum_order"]))

    return curriculum_end_indices


def get_cc_dataset_and_pacing_fn(tokenizer, block_size, total_steps,
                           spoken_first=True, pacing_fn_name="log",
                           start_percent=0.1, end_percent=0.9,
                            starting_difficulty=0.1, max_difficulty=1, growth_rate_c=10):
    """
    Loads the dataset and preprocesses it for curriculum learning.
    :param tokenizer (PreTrainedTokenizer): instantiated tokenizer object
    :param block_size (int): maximum input length for the model
    :param spoken_first (bool): whether to put spoken datasets first ordering or grammatical datasets first ordering
    :param pacing_fn_name (str): the name of the pacing function to use
    :param start_percent (float): the percentage of steps from the total number of steps that have been taken before we begin increasing the data difficulty
    :param end_percent (float): the percentage of steps from the total number of steps that have been taken after which we stop increasing the data difficulty.
    :param starting_difficulty (float): the starting difficulty of the dataset as a percentile of the dataset's difficulty. A value of 0.2 means that initially, we sample from the bottom 20% difficult examples.
    #Updated to calculate dynamically?
    :param max_difficulty (float): the maximum difficulty of the dataset as a percentile of the dataset's difficulty. A value of 1.0 means that the maximum difficulty we can sample is the maximum difficulty in the dataset.
    :return:
        train_dataset (Dataset): the training dataset
        val_dataset (Dataset): the validation dataset
        train_curriculum_end_indices (Dict[str, int]): the end indices of each curriculum in the training dataset, to use for sampling later
        val_curriculum_end_indices (Dict[str, int]): the end indices of each curriculum in the validation dataset, to use for sampling later
        pacing_fn (Callable[[int], float]): the pacing function to use
    """

    dataset = load_cc_dataset(dataset_name='cambridge-climb/BabyLM',
                           dataset_subconfig='original_strict_small',
                           load_from_hf=True)
    print("original data size", len(dataset["train"]))
    #hardcoding include_punctuation, join_sentences, dataset_subconfig for now
    data_preprocessor = DatasetPreprocessor(tokenizer=tokenizer,
                                            max_input_length=block_size,
                                            include_punctuation=True,
                                            join_sentences=True,
                                            dataset_subconfig="original_strict_small",
                                            spoken_first=spoken_first)

    train_dataset = dataset["train"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["train"].column_names,
    )
    print("processed data size", len(train_dataset))
    val_dataset = dataset["validation"].map(
        data_preprocessor,
        batched=True,
        num_proc=64,
        remove_columns=dataset["validation"].column_names,
    )

    train_dataset = train_dataset.sort("curriculum_order")
    val_dataset = val_dataset.sort("curriculum_order")

    train_curriculum_end_indices = get_curriculum_end_indices(train_dataset)
    val_curriculum_end_indices = get_curriculum_end_indices(val_dataset)

    print(f"setting pacing function to parameters: pacing_fn_name={pacing_fn_name}, total_steps={total_steps}, "
          f"start_percent={start_percent}, end_percent={end_percent}, "
          f"starting_difficulty={starting_difficulty}, max_difficulty={max_difficulty}, growth_rate_factor={growth_rate_c}"
          )

    pacing_fn = get_pacing_fn(pacing_fn_name, total_steps,
                              start_percent, end_percent,
                              starting_difficulty=starting_difficulty,
                              max_difficulty=max_difficulty,
                              growth_rate_c=growth_rate_c
                              )

    return train_dataset, val_dataset, train_curriculum_end_indices, val_curriculum_end_indices, pacing_fn


def convert_pacing_fn_to_max_difficulty(pacing_fn, current_iter, difficulties_list, implementation_type="default"):

    """
    Returns the maximum difficulty as a index of dataset difficulty as defined in DATASET_ORDER_MAP

    :param pacing_fn (callable): the pacing function to use
    :param current_iter (int): the current iteration
    :param difficulties_list (np array): the dataset to access the curriculum order to get nth percentile from
    :param implementation_type (str): "default" or "absolute" the implementation type to use, default or custom

    default is from CLIMB paper where max difficulty is used to select difficulty score
    from 1-6 based on percentile of data, because number of training examples of each difficulty level is not equal
    Alternatively, we could just use the percentile to convert it absolutely to a difficulty score
    irrespective of how data is distributed

    Returns:
        * max_difficulty (int): the maximum difficulty index
    """

    if implementation_type == "default":
        max_difficulty_percentile = pacing_fn(current_iter)
        max_difficulty = max(int(np.percentile(difficulties_list, max_difficulty_percentile * 100)), 1)

    elif implementation_type == "absolute":
        #hardcoded as 6, maybe change later if needed
        max_difficulty_percentile = pacing_fn(current_iter)
        max_difficulty = max(int(6*max_difficulty_percentile), 1)

    return max_difficulty



