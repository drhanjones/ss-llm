import sys
import torch
import os
from transformers import AutoTokenizer, GPT2Tokenizer
import pickle
import pandas as pd
import platform
from torch.nn import functional as F
import tqdm

if "pop-os" in platform.node():
    TOKENIZER_ROOT = r"/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data"
    OUT_ROOT = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump'
else:
    #raise NotImplementedError("Add the path for the server here")
    OUT_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/output_dump'
    TOKENIZER_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/data'

def load_model(out_dir, device):
    """
    Loads a pre-trained GPT model from a checkpoint file.

    Args:
        out_dir (str): The directory where the checkpoint file is located.
        device (torch.device): The device to load the model onto.

    Returns:
        GPT: The loaded GPT model.

    Raises:
        FileNotFoundError: If the checkpoint file is not found.
    """
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Loading model from {ckpt_path}")
    # NANOGPT_ROOT = str(Path(__file__).parents[4])

    # Add if condition to check if inside server and if is, then add the path correctly. Default is local for now
    if "pop-os" in platform.node():
        NANOGPT_ROOT = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT'  # Edit later to be dynamic
    else:
        NANOGPT_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT'
    sys.path.append(NANOGPT_ROOT)
    from model import GPT, GPTConfig

    checkpoint = torch.load(ckpt_path, map_location=device)

    # Backward compatibility for new model args for QKV and FFW Adjustments
    if checkpoint["model_args"].get("wm_decay_length", None) is None:
        # wm_decay_length = block_size
        checkpoint["model_args"]["wm_decay_length"] = checkpoint["model_args"]["block_size"]
    # Setting head size as 3 times n_embd if not set already
    if checkpoint['model_args'].get('head_size_qkv', None) is None:
        checkpoint['model_args']['head_size_qkv'] = checkpoint['model_args']['n_embd']

    if checkpoint["model_args"].get("ffw_dim", None) is None:
        checkpoint["model_args"]["ffw_dim"] = 4 * checkpoint["model_args"]["n_embd"]

    # print(checkpoint['model_args'])
    gptconf = GPTConfig(**checkpoint['model_args'])

    load_model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    load_model.load_state_dict(state_dict)
    load_model.eval()

    load_model = load_model.to(device)

    return load_model


def load_tokenizer(data_dir):
    """
    Load tokenizer for natural stories evaluation.

    Args:
        data_dir (str): The directory path where the tokenizer data is stored.

    Returns:
        tokenizer (Tokenizer): The loaded tokenizer object.

    Raises:
        NotImplementedError: If stoi/itos is not supported or found.

    """
    meta_path = os.path.join(data_dir, 'meta.pkl')
    load_meta = os.path.exists(meta_path)
    if load_meta:
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        if meta.get("custom_tokenizer", False):
            print(f"Loading custom tokenizer from {data_dir}")
            tokenizer = AutoTokenizer.from_pretrained(data_dir, use_fast=False)
        else:
            if meta.get("stoi", False):
                raise NotImplementedError("stoi/itos not supported yet")
            else:
                raise NotImplementedError("No stoi/itos found")
    else:
        print("No meta.pkl found, using default GPT-2 tokenizer")
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    if not tokenizer.eos_token:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"  # Add if needed?
    return tokenizer


def load_model_tokenizer(out_dir, data_dir, device="cuda"):
    model = load_model(out_dir, device)
    tokenizer = load_tokenizer(data_dir)
    return model, tokenizer



def load_RT_data(rt_root=r'naturalstories_RTS'):
    """
    Load the processed RT data from the RT_root directory
    :param rt_root:
    :return: processsed_RTs, processed_wordinfo, all_stories, where processed RTs are at WorkerId level ...(fill)

    """

    pr_RTs = pd.read_csv(os.path.join(rt_root, 'processed_RTs.tsv'), sep='\t')
    # column Item represents the story number, zone is word analogue to word number in the story. Sort by Item and Zone to get the word order in the story
    pr_RTs = pr_RTs.sort_values(by=['item', 'WorkerId', 'zone'])

    pr_wi = pd.read_csv(os.path.join(rt_root, 'processed_wordinfo.tsv'), sep='\t')
    pr_wi = pr_wi.sort_values(by=['item', 'zone'])

    all_st = pd.read_csv(os.path.join(rt_root, 'all_stories.tok'), sep='\t')
    all_st = all_st.sort_values(by=['item', 'zone'])

    return pr_RTs, pr_wi, all_st


def extract_stories_from_df(stories_df):
    """
    Extract stories from the dataframe with id as key and story as value
    :param stories_df:
    :return: stories: Dictionary with story id as key and story as value
    """
    stories = {}
    story_ids = stories_df["item"].unique()
    for story_id in story_ids:
        story = stories_df[stories_df["item"] == story_id]
        story_text = story.sort_values(by=['zone'])['word'].str.cat(sep=' ')
        stories[story_id] = story_text

    return stories


def tokenize_story(story, tokenizer):
    """
    Tokenize the story using the tokenizer
    :param story: -> str
    :param tokenizer: -> tokenizer
    :return: tokenized_story -> tensor
    """
    tokenized_story = tokenizer.encode(story, return_tensors='pt')
    #since passing only one story, remove the batch dimension
    tokenized_story = tokenized_story.squeeze()
    return tokenized_story


def multitoken_wordmap(token_list, story_id, story_df, tokenizer):
    """
    Create a mapping of words to tokens for a given story because 1 word can be multiple tokens
    :param token_list:
    :param story_id:
    :param story_df:
    :param tokenizer:
    :return: token_map: List of dictionaries with word, zone, tokens, item
    """
    word_list = story_df[story_df['item'] == story_id][['word', 'zone']].to_dict('records')
    word_list = sorted(word_list, key=lambda x: x['zone'])

    token_map = []
    token_index = 0
    for i, word_row in enumerate(word_list):
        word = word_row['word']
        zone = word_row['zone']
        #since 1 word can be multiple tokens, we need to keep track of the tokens that make up the word
        decode_list = []
        while True:
            if token_index >= len(token_list):
                break
            decode_list.append(token_list[token_index].item())
            decoded_words = tokenizer.decode(decode_list).strip()
            token_index += 1
            if word.lower() == decoded_words:
                break

        token_map.append({"item": story_id, "zone": zone, "word": word, "tokens": decode_list})
    return token_map


def return_surprisals(model, token_list, device='cuda'):
    if len(token_list)>model.config.block_size:
        token_list = token_list[-model.config.block_size:]
    token_tensor = torch.tensor(token_list).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(token_tensor[:, :-1], token_tensor[:, 1:])
    probs = F.log_softmax(logits, dim=-1)
    token_logprob = probs[0, -1, token_tensor[0, -1]].item()
    return -token_logprob


def get_model_surprisals(model, tokenized_story_df, story_id, tokenizer):
    tokenized_story = tokenized_story_df[tokenized_story_df['item'] == story_id].to_dict('records')
    tokenized_story = sorted(tokenized_story, key=lambda x: x['zone'])
    #Prepend the bos token
    logits_input_list = [tokenizer.bos_token_id]

    for word_row in tqdm.tqdm(tokenized_story, leave=False):
        tokens = word_row['tokens']
        if len(tokens) == 1:
            logits_input_list.append(tokens[0])
            word_surprisal = return_surprisals(model, logits_input_list)
        else:
            word_surprisal = 0
            for token in tokens:
                logits_input_list.append(token)
                word_surprisal += return_surprisals(model, logits_input_list)

        word_row['surprisal'] = word_surprisal

    return tokenized_story


if __name__ == '__main__':
    pass