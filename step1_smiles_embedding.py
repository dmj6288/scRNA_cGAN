import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
from concurrent.futures import ThreadPoolExecutor
import pickle as pkl

# Use the correct tokenizer and model for ChemBERTa
tokenizer = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = RobertaModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")


smiles = pd.read_csv("../ZafrensData/smiles.csv")

columns_to_concat = ['control_rx_id', 'bb1_id', 'bb2_id', 'bb3_id', 'bb4_id']

smiles['sample'] = smiles[columns_to_concat].astype(str).agg('_'.join, axis=1)

# Example SMILES strings
smiles_data = smiles['SMILES'].tolist()[7000:]

# Function to get embeddings for a single SMILES
def get_single_embedding(smiles):
    # Tokenize the SMILES string
    inputs = tokenizer(smiles, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    # Get embeddings from the model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the [CLS] token embedding
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.squeeze().numpy()

# Parallel function to get embeddings for all SMILES
def get_chemberta_embeddings_parallel(smiles_list, num_threads=4):
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Map the get_single_embedding function to each SMILES in parallel
        embeddings = list(executor.map(get_single_embedding, smiles_list))
    return embeddings

# Set the number of threads based on your system's resources
num_threads = torch.get_num_threads() if torch.cuda.is_available() else 16

# Get embeddings in parallel
embeddings = get_chemberta_embeddings_parallel(smiles_data, num_threads=num_threads)

# Convert to a DataFrame for visualization or further processing
embeddings_df = pd.DataFrame(embeddings)
print(embeddings_df.head(10))

with open("../ZafrensData/embedded_smiles/embedded_smiles_2.pkl", "wb") as file:
    pkl.dump(embeddings_df, file)
