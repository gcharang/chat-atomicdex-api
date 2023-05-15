from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
from weaviate.util import get_valid_uuid
import pandas as pd
import tiktoken
from uuid import uuid4

from tqdm import tqdm

import json
import os
import re
import zipfile
import time
import datetime

from urllib.request import urlopen
from io import BytesIO

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

weaviate_class_name = "Code"
weaviate_client = weaviate.Client(url="http://weaviate:8080")
weaviate_client.schema.delete_class(weaviate_class_name)
print(f"Old index on weaviate: {weaviate_class_name} was deleted")
weaviate_client.schema.delete_all()
weaviate_client.schema.get()
weaviate_schema = {
    "classes": [
        {
            "class": weaviate_class_name,
            "description": "A class called code snippet",
            "vectorIndexConfig": {
                "distance": "cosine",
            },
            "vectorizer": "text2vec-openai",
            "moduleConfig": {
                "text2vec-openai": {
                "vectorizeClassName": False,
                "model": "ada",
                "modelVersion": "002",
                "type": "text"
                }
            },
            "properties": [
                {
                "dataType": ["text"],
                "description": "Content that will be vectorized",
                "moduleConfig": {
                    "text2vec-openai": {
                    "skip": False,
                    "vectorizePropertyName": False
                    }
                },
                "name": "page_content"
                },
            ]
            }
        ]
    }

weaviate_client.schema.create(weaviate_schema)
print("Weaviate schema created")

def json_serializable(value):
            if isinstance(value, datetime.datetime):
                return value.isoformat()
            return value
        
items_processed_to_track_embedding_progress = 0

def configure_batch(client: weaviate_client, batch_size: int, batch_target_rate: int):
    """
    Configure the weaviate client's batch so it creates objects at `batch_target_rate`.

    Parameters
    ----------
    client : Client
        The Weaviate client instance.
    batch_size : int
        The batch size.
    batch_target_rate : int
        The batch target rate as # of objects per second.
    """

    def callback(batch_results: dict) -> None:
        # with open('output.txt', 'w') as f:
        #     json.dump(batch_results, f)
        # you could print batch errors here
        global items_processed_to_track_embedding_progress
        items_processed_to_track_embedding_progress = items_processed_to_track_embedding_progress + len(batch_results)
        pbar2.n = 0
        pbar2.refresh()
        pbar2.update(items_processed_to_track_embedding_progress)
        time_took_to_create_batch = batch_size * (client.batch.creation_time/client.batch.recommended_num_objects)
        time_to_sleep=max(batch_size/batch_target_rate - time_took_to_create_batch + 1, 0)
        print(f"Sleeping for {time_to_sleep} seconds")
        time.sleep(time_to_sleep)

    client.batch.configure(
        batch_size=batch_size,
        timeout_retries=5,
        callback=callback,
        dynamic=True
    )

def zipfile_from_github():
    http_response = urlopen(os.environ['REPO_BRANCH_ZIP_URL'])
    zf = BytesIO(http_response.read())
    print(f"downloaded zipfile from github: {os.environ['REPO_BRANCH_ZIP_URL']}")
    return zipfile.ZipFile(zf, 'r')

encoder = tiktoken.get_encoding(os.environ['ENCODING'])

splitter = RecursiveCharacterTextSplitter(
    chunk_size=int(os.environ['CHUNK_SIZE']),
    chunk_overlap=int(os.environ['CHUNK_OVERLAP'])
    )

total_tokens, corpus_summary = 0, []
file_texts, metadatas = [], []
with zipfile_from_github() as zip_ref:
    zip_file_list = zip_ref.namelist()
    print("reading file texts and calculating total number of tokens")
    pbar = tqdm(zip_file_list, desc=f'Total tokens: 0')
    for file_name in pbar:
        if (file_name.endswith('/') or 
            any(f in file_name for f in ['.DS_Store', '.gitignore']) or 
            any(file_name.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])
        ):
            continue
        else:
            with zip_ref.open(file_name, 'r') as file:
                file_contents = str(file.read())
                file_name_trunc = re.sub(r'^[^/]+/', '', str(file_name))
                                
                n_tokens = len(encoder.encode(file_contents))
                total_tokens += n_tokens
                corpus_summary.append({'file_name': file_name_trunc, 'n_tokens': n_tokens})

                file_texts.append(file_contents)
                metadatas.append({'document_id': file_name_trunc})
                pbar.set_description(f'Total tokens: {total_tokens}')
                
pd.DataFrame.from_records(corpus_summary).to_csv(os.environ['CORPUS_SUMMARY_OUTPUT_FILE_PATH'], index=False)
print(f"wrote corpus summary to: {os.environ['CORPUS_SUMMARY_OUTPUT_FILE_PATH']}")

print(f"Cost to embed: ${(total_tokens * 0.0004)/1000}")
print("Splitting files into chunks...")
split_documents = splitter.create_documents(file_texts, metadatas=metadatas)
print("...done")

before = time.time()
num_chunks_to_embed=len(split_documents)
ids = list(range(num_chunks_to_embed))

print(f"embedding {num_chunks_to_embed} items with text-ada-002 and storing them in weaviate...")


pbar2 = tqdm(total=num_chunks_to_embed, desc=f"chunks embedded")

configure_batch(client=weaviate_client,batch_size=1000,batch_target_rate=(3500/70))

with weaviate_client.batch as batch:
    ids = []
    for i, doc in enumerate(split_documents):
        data_properties = {
            "page_content": doc.page_content,
            "document_id": doc.metadata["document_id"]
        }

        _id = get_valid_uuid(uuid4())

        batch.add_data_object(
            data_object=data_properties,
            class_name=weaviate_class_name,
            uuid=_id,
        )
        ids.append(_id)

pbar2.close()

