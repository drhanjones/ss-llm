import pandas as pd
import sqlite3
import numpy as np
import os
import platform 
import torch
import pickle
from transformers import GPT2Tokenizer, AutoTokenizer
from torch.nn import functional as F
import sys
from tqdm import tqdm
tqdm.pandas()


if "pop-os" in platform.node():
    ROOT = r"/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/"
else:
    ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/'
    
TOKENIZER_ROOT = os.path.join(ROOT, "data")
OUT_ROOT = os.path.join(ROOT, "output_dump")
RESULTS_ROOT = os.path.join(ROOT, "results")

SQL_DB = os.path.join(RESULTS_ROOT, "results.db")


def create_connection_cursor(db_file):
    """
    Create a database connection to the SQLite database specified by the db_file

    Args:
        db_file (str): database file

    Returns:
        Connection object or None
    """
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    return conn, c


def load_model(model_id, c, device="cuda"):
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


    c.execute("SELECT OutputFolderName FROM Model WHERE ModelID=?", (model_id,))
    out_fol = c.fetchone()[0]

    try:    
        out_dir = os.path.join(OUT_ROOT, out_fol)
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}, trying in external drive")
    except FileNotFoundError as e:
        #print(e)
        NEW_OUT_ROOT = "/media/abishekthamma/Backup Plus/Projects/masters_thesis/ss-llm/nanoGPT/output_dump"
        out_dir = os.path.join(NEW_OUT_ROOT, out_fol)
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}, unable to load model")

    #print(f"Loading model from {ckpt_path}")
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

    #NOTE! THIS FILE INVERTS THE MASK OF THE MODEL DURING INFERENCE

    checkpoint['model_args']['wm_mask'] = not checkpoint['model_args']['wm_mask']

    if checkpoint['model_args']["wm_mask"] == True:
        checkpoint['model_args']["wm_decay_rate"] = 2
        checkpoint['model_args']["wm_decay_type"] = "exponential_2"
        checkpoint['model_args']["wm_decay_echoic_memory"] = 10
        
    # print(checkpoint['model_args'])
    gptconf = GPTConfig(**checkpoint['model_args'])

    load_model = GPT(gptconf)

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        #print(k)

        #Removing Bias part from state_dict because non masked models use Flash Attention becaUSE OF code in model.py and that means that the bias variable doesn't need to be manually defined 
        if checkpoint['model_args']["wm_mask"] == False:
            if k.endswith('.bias'):
                state_dict.pop(k)   
                continue
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    if checkpoint['model_args']["wm_mask"] == True:
        #Flipping a non masked model to a masked model means it doesn't have the bias variable that is manually defined when disabling flash attention and load state dict wil cry about it
        load_model.load_state_dict(state_dict, strict=False)
    else:
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


def get_context_list(token_string, context_window, tokenizer):

    bos_token = tokenizer.bos_token_id

    context_list = [bos_token]

    if token_string:
        context_list.extend([int(i) for i in token_string.split(",")])
    else:
        return context_list
    
    if len(context_list) > context_window:
        context_list = context_list[-context_window:]

    return context_list


def get_surprisal_inputs(model_id, story_id=None, model_path=None, tokenizer=None):
    conn, c = create_connection_cursor(SQL_DB)

    #For a given model, get its tokenizer id and context window size
    c.execute("SELECT TokenizerID, BlockSize FROM Model WHERE ModelID=?", (model_id,))
    model_row = c.fetchone()
    if model_row is None:
        print("Model ID not found in the database")
        return None
    
    tokenizer_id = model_row[0]
    context_window = model_row[1]
    story_id_part = ""
    if story_id:
        story_id_part = f"AND StoryID={story_id}"


    #Query to get the context window of n or less words before the word given a story id
    query_fin = f"""  WITH    StoryWordRank 
                        AS  (SELECT  Story.StoryWordID,  
                                    Story.StoryID, 
                                    TokenizedStory.TokenValue,
                                    row_number() 
                                        OVER (
                                            PARTITION BY Story.CorpusID, Story.StoryID) AS 
                                    story_token_rank
                            FROM    TokenizedStory
                                    JOIN Story 
                                        On TokenizedStory.StoryWordID = Story.StoryWordID
                            WHERE TokenizedStory.TokenizerID = {tokenizer_id} {story_id_part} AND Story.CorpusID = 1) 
                    SELECT  StoryWordRank.StoryWordID,
                            StoryWordRank.StoryID, 
                            StoryWordRank.TokenValue,
                            
                            group_concat(TokenValue) 
                                OVER (
                                    PARTITION BY StoryID 
                                    ORDER BY story_token_rank ROWS BETWEEN {context_window} PRECEDING AND 1 PRECEDING)  
                                    
                                As 
                                ContextWindow 
                    FROM StoryWordRank """
    
    #c.execute(query_fin)
    #Query returns StoryID, Word, ContextWindow
    #tokenwise_context_story = c.fetchall()
    tokenwise_context_story = pd.read_sql_query(query_fin, con=conn)
    conn.close()

    tokenwise_context_story["ContextWindow"] = tokenwise_context_story["ContextWindow"].apply(lambda x: get_context_list(x, context_window, tokenizer))
    tokenwise_context_story["TokenValue"] = tokenwise_context_story["TokenValue"].apply(lambda x: int(x))

    return tokenwise_context_story

def df_calculate_surprisal(model, context_window, output_token, device='cuda'):
    """
    Given a model, context window and output token, calculate the surprisal for the output token
    """
    context_window_tensor = torch.tensor(context_window).unsqueeze(0).to(device)
    output_tensor = torch.tensor(output_token).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = model(context_window_tensor)

    probs = F.log_softmax(logits, dim=-1)
    token_logprob = probs[0, -1, output_tensor[0, -1]].item()

    return -token_logprob

def df_calculate_surprisal_batch(model, context_window_df, context_window_length, batch_size = 256, device='cuda'):
    """
    Given a model, context window and output token, calculate the surprisal for the output token. Batch data for different length context windows
    """

    for j in tqdm(range(context_window_length)):
        single_context_window = context_window_df[context_window_df["context_window_length"] == j+1]
        context_window = single_context_window["ContextWindow"].tolist()
        output_token = single_context_window["TokenValue"].tolist()
        
        for i in tqdm(range(0, len(context_window), batch_size), leave=False):
            #print("i is", i, "j is", j)
            context_window_tensor = torch.tensor(context_window[i:i+batch_size]).to(device)
            output_tensor = torch.tensor(output_token[i:i+batch_size]).to(device)

            with torch.no_grad():
                logits, _ = model(context_window_tensor)

            probs = F.log_softmax(logits, dim=-1)
            token_logprob = probs.gather(2, output_tensor.unsqueeze(1).unsqueeze(2))

            if j == 0 and i == 0:
                #print("H1")
                surprisal_tensor = -token_logprob
                #print(surprisal_tensor.shape, surprisal_tensor)
            else:
                #print("H2")
                #print(surprisal_tensor.shape, surprisal_tensor, -token_logprob)
                surprisal_tensor = torch.cat((surprisal_tensor, -token_logprob))

    context_window_df["Surprisal"] = surprisal_tensor.squeeze().tolist()

    return context_window_df


def process_model_story(model_id, conn, c):
    loaded_model = load_model(model_id, c)
    model_tokenizer = c.execute("SELECT TokenizerName FROM Tokenizer WHERE TokenizerID=(SELECT TokenizerID FROM Model WHERE ModelID=?)", (model_id,)).fetchone()
    loaded_tokenizer = load_tokenizer(os.path.join(TOKENIZER_ROOT, model_tokenizer[0]))

    processing_story_df = get_surprisal_inputs(model_id, tokenizer=loaded_tokenizer)
    processing_story_df = processing_story_df.reset_index(drop=True)
    processing_story_df["context_window_length"] = processing_story_df["ContextWindow"].apply(lambda x: len(x))
    processing_story_df["row_id"] = processing_story_df.index
    processing_story_df.sort_values(by=["context_window_length"], inplace=True)
    
    processing_story_df = processing_story_df.reset_index(drop=True)

    processing_story_df = df_calculate_surprisal_batch(loaded_model, processing_story_df, processing_story_df["context_window_length"].max())

    processing_story_df.sort_values(by=["row_id"], inplace=True)
    processing_story_df = processing_story_df.groupby("StoryWordID").agg({"Surprisal": "sum"}).reset_index()
    processing_story_df["ModelID"] = model_id
    processing_story_df = processing_story_df.rename(columns={"Surprisal": "SurprisalScore"})


    return processing_story_df[["ModelID", "StoryWordID", "SurprisalScore"]]

def insert_into_db(processed_model_surprisal_df, conn, c):

    for index, row in processed_model_surprisal_df.iterrows():
        c.execute("INSERT INTO InvertedMaskModelSurprisalScores (ModelID, StoryWordID, SurprisalScore) VALUES (?, ?, ?)", (row["ModelID"], row["StoryWordID"], row["SurprisalScore"]))

    conn.commit()

def get_incomplete_model_ids(conn, c):
    """
    Get model IDs that do not have suprisal scores calculated already (Number of rows for each model ID in ModelSurprisalScores table must be equal to number of rows in Story table)

    Returns:
        List of model IDs that do not have surprisal scores calculated
    """

    expected_row_counts = c.execute("Select Count(*) from Story").fetchone()[0]

    model_ids = c.execute("SELECT ModelID FROM InvertedMaskModelSurprisalScores GROUP BY ModelID HAVING Count(*) != ? ORDER BY ModelID DESC", (expected_row_counts,)).fetchall()

    # model_id_list = pd.read_sql_query("SELECT ModelID FROM ModelSurprisalScores", conn)['ModelID'].unique().tolist()
    #print(len(model_id_list), model_id_list)
    MODEL_LIST_QUERY = '''
    SELECT DISTINCT ModelSurprisalScores.ModelID, Model.OutputFolderName, Model.BatchSize, Model.Dataset, Model.Seed, Model.MaskType FROM ModelSurprisalScores
    JOIN Model on Model.ModelID = ModelSurprisalScores.ModelID
    WHERE Model.NumLayers = 6 AND Model.MaskType="exponential_new" AND Model.MaskDecayRate=2 
    AND Model.ModelID not in (5496427, 8456913)
    ORDER BY Seed, BatchSize, Dataset, MaskType

    '''

    MODEL_LIST_QUERY = '''
    SELECT DISTINCT ModelSurprisalScores.ModelID, Model.OutputFolderName, Model.BatchSize, Model.Dataset, Model.Seed, Model.MaskType FROM ModelSurprisalScores
    JOIN Model on Model.ModelID = ModelSurprisalScores.ModelID
    WHERE Model.NumLayers = 6 AND Model.MaskType="Non" AND Model.CurriculumLearning = 0 AND Model.BatchSize = 32 
    AND Model.ModelID not in (5496427, 8456913)
    ORDER BY Seed, BatchSize, Dataset, MaskType

    '''

    model_id_list = pd.read_sql_query(MODEL_LIST_QUERY, conn)['ModelID'].unique().tolist()
    return model_id_list
    # print(len(model_id_list), model_id_list)
    # already_processed_model_ids = pd.read_sql_query("SELECT ModelID FROM InvertedMaskModelSurprisalScores", conn)['ModelID'].unique().tolist()
    # print(len(already_processed_model_ids), already_processed_model_ids)
    # #return [i[0] for i in model_ids]
    # return list(set(model_id_list) - set(already_processed_model_ids))

def main():
    conn, c = create_connection_cursor(SQL_DB)

    incomplete_model_ids = get_incomplete_model_ids(conn, c)
    print("Processing ", len(incomplete_model_ids), " models")

    for k, model_id in enumerate(incomplete_model_ids):
        model_mask_setting = c.execute("SELECT ModelID, Masking from Model WHERE ModelID=?", (model_id,)).fetchone()
        
        if model_mask_setting is None:
            print("Model ID not found in the database")
            continue
        # if model_mask_setting[1] == 0:
        #     print(f"Model {model_id} is a non-masked model, skipping")
        #     continue

        if model_mask_setting[1] != 0:
            print(f"Model {model_id} is a masked model, which are already processed")
            continue

        try:   
            print("Processing model ", model_id, ", ", k+1, " of ", len(incomplete_model_ids))
            processed_model_surprisal_df = process_model_story(model_id, conn, c)
            insert_into_db(processed_model_surprisal_df, conn, c)
        except FileNotFoundError as e:
            print(e)
            print("Skipping model ", model_id)
            continue
        # except Exception as e:
        #     print(e)
        #     print("Error processing model ", model_id)
        #     continue
        
if __name__ == "__main__":
    main()

    # conn, c = create_connection_cursor(SQL_DB)
    # incomplete_model_ids = get_incomplete_model_ids(conn, c)
    # print("Processing ", len(incomplete_model_ids), " models")

    # for ik in incomplete_model_ids:
    #     model_mask_setting = c.execute("SELECT ModelID, Masking from Model WHERE ModelID=?", (ik,)).fetchone()

    #     if model_mask_setting[1] != 0:
    #         print(f"Model {ik} is a masked model, skipping")
    #         #m1 = load_model(model_mask_setting[0], c)
    #         #print(m1)
    #         #break
    #     else:
    #         print(f"Model {ik} is a non masked model, processing")
    #         m1 = load_model(ik, c)
    #         print(f"Model {ik} loaded")
    #         print(m1)
    #         break

    











# import os
# import pandas as pd
# import platform
# #from NS_eval_utils import load_model_tokenizer, load_RT_data, extract_stories_from_df, tokenize_story, multitoken_wordmap, get_model_surprisals


# import sys
# import torch
# import os
# from transformers import AutoTokenizer, GPT2Tokenizer
# import pickle
# import pandas as pd
# import platform
# from torch.nn import functional as F
# import tqdm

# if "pop-os" in platform.node():
#     TOKENIZER_ROOT = r"/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data"
#     OUT_ROOT = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump'
# else:
#     #raise NotImplementedError("Add the path for the server here")
#     OUT_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/output_dump'
#     TOKENIZER_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/data'

# # def load_model(out_dir, device):
# #     """
# #     Loads a pre-trained GPT model from a checkpoint file.

# #     Args:
# #         out_dir (str): The directory where the checkpoint file is located.
# #         device (torch.device): The device to load the model onto.

# #     Returns:
# #         GPT: The loaded GPT model.

# #     Raises:
# #         FileNotFoundError: If the checkpoint file is not found.
# #     """
# #     ckpt_path = os.path.join(out_dir, 'ckpt.pt')
# #     print(f"Loading model from {ckpt_path}")
# #     # NANOGPT_ROOT = str(Path(__file__).parents[4])

# #     # Add if condition to check if inside server and if is, then add the path correctly. Default is local for now
# #     if "pop-os" in platform.node():
# #         NANOGPT_ROOT = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT'  # Edit later to be dynamic
# #     else:
# #         NANOGPT_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT'
# #     sys.path.append(NANOGPT_ROOT)
# #     from model import GPT, GPTConfig

# #     checkpoint = torch.load(ckpt_path, map_location=device)

# #     # Backward compatibility for new model args for QKV and FFW Adjustments
# #     if checkpoint["model_args"].get("wm_decay_length", None) is None:
# #         # wm_decay_length = block_size
# #         checkpoint["model_args"]["wm_decay_length"] = checkpoint["model_args"]["block_size"]
# #     # Setting head size as 3 times n_embd if not set already
# #     if checkpoint['model_args'].get('head_size_qkv', None) is None:
# #         checkpoint['model_args']['head_size_qkv'] = checkpoint['model_args']['n_embd']

# #     if checkpoint["model_args"].get("ffw_dim", None) is None:
# #         checkpoint["model_args"]["ffw_dim"] = 4 * checkpoint["model_args"]["n_embd"]

# #     # print(checkpoint['model_args'])
# #     gptconf = GPTConfig(**checkpoint['model_args'])

# #     load_model = GPT(gptconf)

# #     state_dict = checkpoint['model']
# #     unwanted_prefix = '_orig_mod.'
# #     for k, v in list(state_dict.items()):
# #         if k.startswith(unwanted_prefix):
# #             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

# #     load_model.load_state_dict(state_dict)
# #     load_model.eval()

# #     load_model = load_model.to(device)

# #     return load_model

# def load_model(model_id, c, device="cuda"):
#     """
#     Loads a pre-trained GPT model from a checkpoint file.

#     Args:
#         out_dir (str): The directory where the checkpoint file is located.
#         device (torch.device): The device to load the model onto.

#     Returns:
#         GPT: The loaded GPT model.

#     Raises:
#         FileNotFoundError: If the checkpoint file is not found.
#     """


#     c.execute("SELECT OutputFolderName FROM Model WHERE ModelID=?", (model_id,))
#     out_fol = c.fetchone()[0]

#     try:    
#         out_dir = os.path.join(OUT_ROOT, out_fol)
#         ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}, trying in external drive")
#     except FileNotFoundError as e:
#         print(e)
#         NEW_OUT_ROOT = "/media/abishekthamma/Backup Plus/Projects/masters_thesis/ss-llm/nanoGPT/output_dump"
#         out_dir = os.path.join(NEW_OUT_ROOT, out_fol)
#         ckpt_path = os.path.join(out_dir, 'ckpt.pt')
#         if not os.path.exists(ckpt_path):
#             raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}, unable to load model")

#     print(f"Loading model from {ckpt_path}")
#     # NANOGPT_ROOT = str(Path(__file__).parents[4])

#     # Add if condition to check if inside server and if is, then add the path correctly. Default is local for now
#     if "pop-os" in platform.node():
#         NANOGPT_ROOT = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT'  # Edit later to be dynamic
#     else:
#         NANOGPT_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT'
#     sys.path.append(NANOGPT_ROOT)
#     from model import GPT, GPTConfig

#     checkpoint = torch.load(ckpt_path, map_location=device)

#     # Backward compatibility for new model args for QKV and FFW Adjustments
#     if checkpoint["model_args"].get("wm_decay_length", None) is None:
#         # wm_decay_length = block_size
#         checkpoint["model_args"]["wm_decay_length"] = checkpoint["model_args"]["block_size"]
#     # Setting head size as 3 times n_embd if not set already
#     if checkpoint['model_args'].get('head_size_qkv', None) is None:
#         checkpoint['model_args']['head_size_qkv'] = checkpoint['model_args']['n_embd']

#     if checkpoint["model_args"].get("ffw_dim", None) is None:
#         checkpoint["model_args"]["ffw_dim"] = 4 * checkpoint["model_args"]["n_embd"]

#     #NOTE! THIS FILE INVERTS THE MASK OF THE MODEL DURING INFERENCE

#     checkpoint['model_args']['wm_mask'] = not checkpoint['model_args']['wm_mask']

#     # print(checkpoint['model_args'])
#     gptconf = GPTConfig(**checkpoint['model_args'])

#     load_model = GPT(gptconf)

#     state_dict = checkpoint['model']
#     unwanted_prefix = '_orig_mod.'
#     for k, v in list(state_dict.items()):
#         print(k)

#         #Removing Bias part from state_dict because non masked models use Flash Attention becaUSE OF code in model.py and that means that the bias variable doesn't need to be manually defined 

#         if k.endswith('.bias'):
#             state_dict.pop(k)   
#             continue
#         if k.startswith(unwanted_prefix):
#             state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

#     load_model.load_state_dict(state_dict)
#     load_model.eval()

#     load_model = load_model.to(device)

#     return load_model


# def load_tokenizer(data_dir):
#     """
#     Load tokenizer for natural stories evaluation.

#     Args:
#         data_dir (str): The directory path where the tokenizer data is stored.

#     Returns:
#         tokenizer (Tokenizer): The loaded tokenizer object.

#     Raises:
#         NotImplementedError: If stoi/itos is not supported or found.

#     """
#     meta_path = os.path.join(data_dir, 'meta.pkl')
#     load_meta = os.path.exists(meta_path)
#     if load_meta:
#         with open(meta_path, 'rb') as f:
#             meta = pickle.load(f)
#         if meta.get("custom_tokenizer", False):
#             print(f"Loading custom tokenizer from {data_dir}")
#             tokenizer = AutoTokenizer.from_pretrained(data_dir, use_fast=False)
#         else:
#             if meta.get("stoi", False):
#                 raise NotImplementedError("stoi/itos not supported yet")
#             else:
#                 raise NotImplementedError("No stoi/itos found")
#     else:
#         print("No meta.pkl found, using default GPT-2 tokenizer")
#         tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

#     if not tokenizer.eos_token:
#         tokenizer.add_special_tokens({"eos_token": "</s>"})
#     if not tokenizer.pad_token:
#         tokenizer.pad_token = tokenizer.eos_token

#     tokenizer.padding_side = "left"  # Add if needed?
#     return tokenizer


# def load_model_tokenizer(out_dir, data_dir, device="cuda"):
#     model = load_model(out_dir, device)
#     tokenizer = load_tokenizer(data_dir)
#     return model, tokenizer



# def load_RT_data(rt_root=r'naturalstories_RTS'):
#     """
#     Load the processed RT data from the RT_root directory
#     :param rt_root:
#     :return: processsed_RTs, processed_wordinfo, all_stories, where processed RTs are at WorkerId level ...(fill)

#     """

#     pr_RTs = pd.read_csv(os.path.join(rt_root, 'processed_RTs.tsv'), sep='\t')
#     # column Item represents the story number, zone is word analogue to word number in the story. Sort by Item and Zone to get the word order in the story
#     pr_RTs = pr_RTs.sort_values(by=['item', 'WorkerId', 'zone'])

#     pr_wi = pd.read_csv(os.path.join(rt_root, 'processed_wordinfo.tsv'), sep='\t')
#     pr_wi.columns = [colname.strip() for colname in pr_wi.columns]
#     pr_wi = pr_wi.sort_values(by=['item', 'zone'])

#     all_st = pd.read_csv(os.path.join(rt_root, 'all_stories.tok'), sep='\t')
#     all_st = all_st.sort_values(by=['item', 'zone'])

#     return pr_RTs, pr_wi, all_st


# def extract_stories_from_df(stories_df):
#     """
#     Extract stories from the dataframe with id as key and story as value
#     :param stories_df:
#     :return: stories: Dictionary with story id as key and story as value
#     """
#     stories = {}
#     story_ids = stories_df["item"].unique()
#     for story_id in story_ids:
#         story = stories_df[stories_df["item"] == story_id]
#         story_text = story.sort_values(by=['zone'])['word'].str.cat(sep=' ')
#         stories[story_id] = story_text

#     return stories


# def tokenize_story(story, tokenizer):
#     """
#     Tokenize the story using the tokenizer
#     :param story: -> str
#     :param tokenizer: -> tokenizer
#     :return: tokenized_story -> tensor
#     """
#     tokenized_story = tokenizer.encode(story, return_tensors='pt')
#     #since passing only one story, remove the batch dimension
#     tokenized_story = tokenized_story.squeeze()

#     return tokenized_story


# def multitoken_wordmap(token_list, story_id, story_df, tokenizer):
#     """
#     Create a mapping of words to tokens for a given story because 1 word can be multiple tokens
#     :param token_list:
#     :param story_id:
#     :param story_df:
#     :param tokenizer:
#     :return: token_map: List of dictionaries with word, zone, tokens, item
#     """
#     word_list = story_df[story_df['item'] == story_id][['word', 'zone']].to_dict('records')
#     word_list = sorted(word_list, key=lambda x: x['zone'])

#     token_map = []
#     token_index = 0
#     for i, word_row in enumerate(word_list):
#         word = word_row['word']
#         zone = word_row['zone']
#         #since 1 word can be multiple tokens, we need to keep track of the tokens that make up the word
#         decode_list = []
#         while True:
#             if token_index >= len(token_list):
#                 break
#             decode_list.append(token_list[token_index].item())
#             decoded_words = tokenizer.decode(decode_list).strip()
#             token_index += 1
#             if word.lower() == decoded_words:
#                 break

#         token_map.append({"item": story_id, "zone": zone, "word": word, "tokens": decode_list})
#     return token_map


# def return_surprisals(model, token_list, device='cuda'):
#     if len(token_list)>model.config.block_size:
#         token_list = token_list[-model.config.block_size:]
#     token_tensor = torch.tensor(token_list).unsqueeze(0).to(device)
#     with torch.no_grad():
#         logits, _ = model(token_tensor[:, :-1], token_tensor[:, 1:]) #probably don't need the second tensor
#     probs = F.log_softmax(logits, dim=-1)
#     token_logprob = probs[0, -1, token_tensor[0, -1]].item()
#     return -token_logprob


# def get_model_surprisals(model, tokenized_story_df, story_id, tokenizer):
#     tokenized_story = tokenized_story_df[tokenized_story_df['item'] == story_id].to_dict('records')
#     tokenized_story = sorted(tokenized_story, key=lambda x: x['zone'])
#     #Prepend the bos token
#     logits_input_list = [tokenizer.bos_token_id]

#     for word_row in tqdm.tqdm(tokenized_story, leave=False):
#         tokens = word_row['tokens']
#         if len(tokens) == 1:
#             logits_input_list.append(tokens[0])
#             word_surprisal = return_surprisals(model, logits_input_list)
#         else:
#             word_surprisal = 0
#             for token in tokens:
#                 logits_input_list.append(token)
#                 word_surprisal += return_surprisals(model, logits_input_list)

#         word_row['surprisal'] = word_surprisal

#     return tokenized_story


# if "pop-os" in platform.node():
#     TOKENIZER_ROOT = r"/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/data"
#     OUT_ROOT = r'/home/abishekthamma/PycharmProjects/masters_thesis/ss-llm/nanoGPT/output_dump'
# else:
#     # raise NotImplementedError("Add the path for the server here")
#     OUT_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/output_dump'
#     TOKENIZER_ROOT = r'/gpfs/home4/athamma/repo/ss-llm/nanoGPT/data'


# def calculate_surprisal_df(out_dir, data_dir):
#     model, tokenizer = load_model_tokenizer(out_dir, data_dir)
#     processed_RTs, processed_wordinfo, all_stories = load_RT_data(rt_root=r'naturalstories_RTS')
#     stories = extract_stories_from_df(all_stories)

#     token_map_df = pd.DataFrame()
#     for i, story in stories.items():
#         tokenized_story_i = tokenize_story(story, tokenizer)
#         token_map = multitoken_wordmap(tokenized_story_i, i, processed_wordinfo, tokenizer)
#         token_map_df = pd.concat([token_map_df, pd.DataFrame(token_map)])

#     story_surprisals_df = pd.DataFrame()
#     for i, story in stories.items():
#         # print(f"Processing story {i}")
#         story_surprisals = get_model_surprisals(model, token_map_df, i, tokenizer)
#         story_surprisals_df = pd.concat([story_surprisals_df, pd.DataFrame(story_surprisals)])

#     #Add model_id to the dataframe
#     #Add out_dir (????) to the dataframe

#     return story_surprisals_df


# def convert_modelname_to_id(inp_model_name):
#     if inp_model_name.startswith('out'):
#         if "curr" in inp_model_name:
#             try:
#                 a_id = inp_model_name.split('-')[-1]
#             except Exception as e:
#                 print(e, inp_model_name)
#                 return None
#         else:
#             try:
#                 a_id = inp_model_name.split('-')[4]
#             except Exception as e:
#                 print(e, inp_model_name)
#                 return None
#         if a_id.isdigit():
#             run_data_key = int(a_id)
#         elif "s" in a_id or "nm" in a_id:
#             run_data_key = int(a_id.split("_")[0])

#     return run_data_key


# if __name__ == "__main__":

#     #model_name = "nanoGPT-2.7B"

#     #data_folder = r'babylm_full_bpe_8k'
#     model_name = 'out-babylm_full_bpe_8k-6x6-mask_lin-5734459_s1337'

#     read_path = "storyword_model_surprisals.csv"
#     surprisal_key_df = pd.read_csv("story_surprisal_keys.csv")

#     model_list = ['out-babylm_full_bpe_100M_8k-6x6-nomask-8569444',
#  'out-babylm_full_bpe_100M_8k-6x6-mask_ee002_em10-8569446']
#     #data_folder_list = []


#     for model_name in model_list:
#         story_surprisals_df_read = pd.read_csv(read_path)
#         if convert_modelname_to_id(model_name) in story_surprisals_df_read['model_id'].values:
#             print(f"Model {model_name} already processed")
#             continue
#         else:
#             print(f"Processing model {model_name}")
#             out_dir = os.path.join(OUT_ROOT, model_name)
#             if "full_bpe_8k" in model_name:
#                 data_folder = r'babylm_full_bpe_8k'
#             elif "wocdes" in model_name:
#                 data_folder = r'babylm_wocdes_full_bpe'
#             elif "babylm_full_bpe_100M_8k" in model_name:
#                 data_folder = r'babylm_full_bpe_100M_8k'
#             else:
#                 data_folder = r'babylm_full_bpe'
#             try:
#                 data_dir = os.path.join(TOKENIZER_ROOT, data_folder)
#                 story_surprisals_df_int = calculate_surprisal_df(out_dir, data_dir)
#                 story_surprisals_df_int["tokenizer"] = data_folder
#                 story_surprisals_df_int = story_surprisals_df_int.merge(surprisal_key_df[["item", "zone", "storyword_UID", "tokenizer"]], on=["item", "zone", "tokenizer"], how="left")
#                 story_surprisals_df_int["model_id"] = convert_modelname_to_id(model_name)
#                 story_surprisals_df_int = story_surprisals_df_int[["model_id", "storyword_UID", "surprisal"]]
#                 story_surprisals_df_read = pd.concat([story_surprisals_df_read, story_surprisals_df_int])

#             except Exception as e:
#                 print(f"Error processing model {model_name}")
#                 print(e)
#                 continue

#             print(f"Writing model {model_name} to file")
#             story_surprisals_df_read.to_csv(read_path, index=False)



