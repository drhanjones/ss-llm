import os
import pandas as pd
import platform
from NS_eval_utils import load_model_tokenizer, load_RT_data, extract_stories_from_df, tokenize_story, multitoken_wordmap, get_model_surprisals

if "pop-os" in platform.node():
    TOKENIZER_ROOT = r"/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data"
    OUT_ROOT = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump'
else:
    # raise NotImplementedError("Add the path for the server here")
    OUT_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/output_dump'
    TOKENIZER_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/data'


def calculate_surprisal_df(out_dir, data_dir):
    model, tokenizer = load_model_tokenizer(out_dir, data_dir)
    processed_RTs, processed_wordinfo, all_stories = load_RT_data(rt_root=r'naturalstories_RTS')
    stories = extract_stories_from_df(all_stories)

    token_map_df = pd.DataFrame()
    for i, story in stories.items():
        tokenized_story_i = tokenize_story(story, tokenizer)
        token_map = multitoken_wordmap(tokenized_story_i, i, processed_wordinfo, tokenizer)
        token_map_df = pd.concat([token_map_df, pd.DataFrame(token_map)])

    story_surprisals_df = pd.DataFrame()
    for i, story in stories.items():
        # print(f"Processing story {i}")
        story_surprisals = get_model_surprisals(model, token_map_df, i, tokenizer)
        story_surprisals_df = pd.concat([story_surprisals_df, pd.DataFrame(story_surprisals)])

    #Add model_id to the dataframe
    #Add out_dir (????) to the dataframe

    return story_surprisals_df


def convert_modelname_to_id(inp_model_name):
    if inp_model_name.startswith('out'):
        if "curr" in inp_model_name:
            try:
                a_id = inp_model_name.split('-')[-1]
            except Exception as e:
                print(e, inp_model_name)
                return None
        else:
            try:
                a_id = inp_model_name.split('-')[4]
            except Exception as e:
                print(e, inp_model_name)
                return None
        if a_id.isdigit():
            run_data_key = int(a_id)
        elif "s" in a_id or "nm" in a_id:
            run_data_key = int(a_id.split("_")[0])

    return run_data_key


if __name__ == "__main__":

    #model_name = "nanoGPT-2.7B"

    #data_folder = r'babylm_full_bpe_8k'
    model_name = 'out-babylm_full_bpe_8k-6x6-mask_lin-5734459_s1337'

    read_path = "storyword_model_surprisals.csv"
    surprisal_key_df = pd.read_csv("story_surprisal_keys.csv")

    model_list = ['out-babylm_full_bpe_8k-6x6-mask_log001-6617787',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047459_s42',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047460_s2347',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047461_s9',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047462_s616',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047464_s46674',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047466_s6747',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047467_s869',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047468_s466',
 'out-babylm_full_bpe_8k-6x6-nomask-curr_log-7047469_s11111']

    #data_folder_list = []


    for model_name in model_list:
        story_surprisals_df_read = pd.read_csv(read_path)
        if convert_modelname_to_id(model_name) in story_surprisals_df_read['model_id'].values:
            print(f"Model {model_name} already processed")
            continue
        else:
            print(f"Processing model {model_name}")
            out_dir = os.path.join(OUT_ROOT, model_name)
            if "full_bpe_8k" in model_name:
                data_folder = r'babylm_full_bpe_8k'
            elif "wocdes" in model_name:
                data_folder = r'babylm_wocdes_full_bpe'
            else:
                data_folder = r'babylm_full_bpe'
            try:
                data_dir = os.path.join(TOKENIZER_ROOT, data_folder)
                story_surprisals_df_int = calculate_surprisal_df(out_dir, data_dir)
                story_surprisals_df_int["tokenizer"] = data_folder
                story_surprisals_df_int = story_surprisals_df_int.merge(surprisal_key_df[["item", "zone", "storyword_UID", "tokenizer"]], on=["item", "zone", "tokenizer"], how="left")
                story_surprisals_df_int["model_id"] = convert_modelname_to_id(model_name)
                story_surprisals_df_int = story_surprisals_df_int[["model_id", "storyword_UID", "surprisal"]]
                story_surprisals_df_read = pd.concat([story_surprisals_df_read, story_surprisals_df_int])
            except Exception as e:
                print(f"Error processing model {model_name}")
                print(e)
                continue

            print(f"Writing model {model_name} to file")
            story_surprisals_df_read.to_csv(read_path, index=False)



