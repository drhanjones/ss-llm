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
        print(e)
        NEW_OUT_ROOT = "/media/abishekthamma/Backup Plus/Projects/masters_thesis/ss-llm/nanoGPT/output_dump"
        out_dir = os.path.join(NEW_OUT_ROOT, out_fol)
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}, unable to load model")

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
                            WHERE TokenizedStory.TokenizerID = {tokenizer_id} {story_id_part} AND Story.CorpusID = 2) 
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

    # expected_row_counts = c.execute("Select Count(*) from Story").fetchone()[0]

    # model_ids = c.execute("SELECT ModelID FROM InvertedMaskModelSurprisalScores GROUP BY ModelID HAVING Count(*) != ? ORDER BY ModelID DESC", (expected_row_counts,)).fetchall()

    # model_id_list = pd.read_sql_query("SELECT ModelID FROM ModelSurprisalScores", conn)['ModelID'].unique().tolist()
    # #print(len(model_id_list), model_id_list)
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

    #return [i[0] for i in model_ids]
    return model_id_list

def main():
    conn, c = create_connection_cursor(SQL_DB)

    incomplete_model_ids = get_incomplete_model_ids(conn, c)
    print("Processing ", len(incomplete_model_ids), " models")

    for k, model_id in enumerate(incomplete_model_ids):
        model_mask_setting = c.execute("SELECT ModelID, Masking from Model WHERE ModelID=?", (model_id,)).fetchone()

        if model_mask_setting is None:
            print(f"Model {model_id} not found in the database")
            continue
        if model_mask_setting[1] != 0:
            print(f"Model {model_id} is a masked model, skipping")
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

    #     if model_mask_setting[1] == 0:
    #         print(f"Model {ik} is a non-masked model, skipping")
    #         #m1 = load_model(model_mask_setting[0], c)
    #         #print(m1)
    #         #break
    #     else:
    #         print(f"Model {ik} is a masked model, processing")
    #         m1 = load_model(ik, c)
    #         print(f"Model {ik} loaded")
    #         print(m1)
    #         break

    


