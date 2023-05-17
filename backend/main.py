import os
import threading
import queue

from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import OpenAIEmbeddings
import weaviate
import json


import openai
import tiktoken
import os

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:5555",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    text: str
    sender: str

class ContextSystemMessage(BaseModel):
    system_message: str
    
class Chat(BaseModel):
    messages: list[dict]

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

def format_context(docs):
    context = '\n\n'.join([f'From file {str(d["document_id"])}:\n' + str(d["page_content"]) for d in docs])
    return context

def format_query(query, context):
    return f"""Relevant context: {context}
    
    {query}"""
    
weaviate_class_name = "Code"
weaviate_client = weaviate.Client(url="http://weaviate:8080")  

def embedding_search(query, k):
    openai.api_key = os.environ['OPENAI_API_KEY']
    model="text-embedding-ada-002"
    oai_resp = openai.Embedding.create(input = [query], model=model)
    oai_embedding = oai_resp['data'][0]['embedding']
    result = (
    weaviate_client.query
    .get(weaviate_class_name, ["page_content", "document_id"])
    .with_hybrid(query,alpha=0.5, vector=oai_embedding)
    .with_limit(k)
    .do()
    )
        #.with_additional(['certainty'])

        # .with_near_vector({
    #     "vector": oai_embedding,
    #     "certainty": 0.7
    # })
    #print(result)
    
    data = result["data"]["Get"][weaviate_class_name]
    
    # Sort the list by the 'certainty' value in the '_additional' dictionary
    #sorted_list = sorted(data, key=lambda x: x['_additional']['certainty'], reverse=True)

    # Create a new list containing only the 'document_id' and 'page_content' of each dictionary
    formatted_result = [{'document_id': d['document_id'], 'page_content': d['page_content']} for d in data]
    #print(formatted_result)
    return formatted_result

@app.get("/health")
def health():
    return "OK"

@app.post("/system_message", response_model=ContextSystemMessage)
def system_message(query: Message):
    
    hyde_prompt = """What is HyDE?
    
    Given a query, HyDE first zero-shot instructs a large language model (e.g. ChatGPT (yourself :))) to generate a hypothetical document. The document captures relevance patterns but is unreal and may contain false details. Then, an unsupervised contrastively learned encoder (e.g.Contriever) encodes the document into an embedding vector. This vector identifies a neighborhood in the corpus embedding space, where similar real documents are retrieved based on vector similarity. This second step ground the generated document to the actual corpus, with the encoder's dense bottleneck filtering out the incorrect details.
    
    At a high level, HyDE is an embedding technique that takes queries, generates a hypothetical answer, and then embeds that generated document to then use the resulting vector to cosine-similarity search a pre-generated corpus of real documents to find the documents that will most likely contain the answer to the given queries.
    
    What is the AtomicDEX API core?
    
    The AtomicDEX API core is open-source [atomic-swap](https://komodoplatform.com/en/academy/atomic-swaps/) software for seamless, decentralised, peer to peer trading between almost every blockchain asset in existence. This software works with propagation of orderbooks and swap states through the [libp2p](https://libp2p.io/) protocol and uses [Hash Time Lock Contracts (HTLCs)](https://en.bitcoinwiki.org/wiki/Hashed_Timelock_Contracts) for ensuring that the two parties in a swap either mutually complete a trade, or funds return to thier original owner. There is no 3rd party intermediary, no proxy tokens, and at all times users remain in sole possession of their private keys. The AtomicDEX-API core is written using the rust programming language. A [well documented API](https://developers.komodoplatform.com/basic-docs/atomicdex/introduction-to-atomicdex.html) offers simple access to the underlying services using simple language agnostic JSON structured methods and parameters such that users can communicate with the core in a variety of methods such as [curl](https://developers.komodoplatform.com/basic-docs/atomicdex-api-legacy/buy.html) in CLI, or fully functioning [desktop and mobile applications](https://atomicdex.io/) like [AtomicDEX Desktop](https://github.com/KomodoPlatform/atomicDEX-Desktop).
    
    You are the hypothetical document generator (HDG) in the HyDE method to answer questions about the contents of the AtomicDEX API core repository.
    The HDG is free to use outside context along with what is stated prior about the AtomicDEX API, to generate a hypothetical document.
    The HDG doesn't generate hypothetical file names/locations. 
    The HDG always respond's with the hypothetical content that might be present in one of the files of the AtomicDEX API repository which can answer the user's query.
    The HDG doesn't add any extra fluff or natural language if it isn't necessary.
    """

    messages = [{"role": "system","content":hyde_prompt}, {"role": "user","content":query.text}]
    openai.api_key = os.environ['OPENAI_API_KEY']
    hyde_completion = openai.ChatCompletion.create(
        model=os.environ['MODEL_NAME'],
        temperature=1, #float(os.environ['TEMPERATURE']),
        messages=messages
    )
    hyde_response = hyde_completion.choices[0]["message"]["content"]
    print(hyde_response)
    
    
    docs = embedding_search(hyde_response, k=int(os.environ['NUM_RELEVANT_DOCS']))
    context = format_context(docs)

    prompt = """Given the following context and code, answer the following question. Do not use outside context, and do not assume the user can see the provided context. Try to be as detailed as possible and reference the components that you are looking at. Keep in mind that these are only code snippets, and more snippets may be added during the conversation.
    Do not generate code, only reference the exact code snippets that you have been provided with. If you are going to write code, make sure to specify the language of the code. For example, if you were writing Rust, you would write the following:

    ```rust
    <rust code goes here>
    ```
    
    Now, here is the relevant context: 

    Context: The AtomicDEX API core is open-source [atomic-swap](https://komodoplatform.com/en/academy/atomic-swaps/) software for seamless, decentralised, peer to peer trading between almost every blockchain asset in existence. This software works with propagation of orderbooks and swap states through the [libp2p](https://libp2p.io/) protocol and uses [Hash Time Lock Contracts (HTLCs)](https://en.bitcoinwiki.org/wiki/Hashed_Timelock_Contracts) for ensuring that the two parties in a swap either mutually complete a trade, or funds return to thier original owner. There is no 3rd party intermediary, no proxy tokens, and at all times users remain in sole possession of their private keys. The AtomicDEX-API core is written using the rust programming language. A [well documented API](https://developers.komodoplatform.com/basic-docs/atomicdex/introduction-to-atomicdex.html) offers simple access to the underlying services using simple language agnostic JSON structured methods and parameters such that users can communicate with the core in a variety of methods such as [curl](https://developers.komodoplatform.com/basic-docs/atomicdex-api-legacy/buy.html) in CLI, or fully functioning [desktop and mobile applications](https://atomicdex.io/) like [AtomicDEX Desktop](https://github.com/KomodoPlatform/atomicDEX-Desktop). {context}
    """
    encoding_name = os.environ['ENCODING']
    encoding = tiktoken.get_encoding(encoding_name)
    system_message_w_context = prompt.format(context=context)
    if len(encoding.encode(system_message_w_context)) > (int(os.environ['MAX_TOKENS']) - int(os.environ['MAX_HUMAN_TOKENS'])):
        chars_to_cut = (len(encoding.encode(system_message_w_context)) - (int(os.environ['MAX_TOKENS']) - int(os.environ['MAX_HUMAN_TOKENS']))) * 4
        system_message_w_context_truncated = system_message_w_context[:-chars_to_cut]
        return {'system_message': system_message_w_context_truncated}    
        
    return {'system_message': prompt.format(context=context)}

@app.post("/chat_stream")
async def chat_stream(chat: List[Message]):
    model_name = os.environ['MODEL_NAME']
    encoding_name = os.environ['ENCODING']

    def llm_thread(g, prompt):
        try:
            llm = ChatOpenAI(
                model_name=model_name,
                verbose=True,
                streaming=True,
                callback_manager=CallbackManager([ChainStreamHandler(g)]),
                temperature=os.environ['TEMPERATURE'],
                openai_api_key=os.environ['OPENAI_API_KEY'],
                openai_organization=os.environ['OPENAI_ORG_ID']
            )

            encoding = tiktoken.get_encoding(encoding_name)
            system_message, latest_query = [chat[0].text, chat[-1].text]
            # the system message gets NUM_RELEVANT_DOCS new docs. Only include NUM_RELEVANT_FOLLOWUP_DOCS more for new queries
            
            if len(chat) > 2 or not latest_query.startswith("continue"):                
                keep_messages = [system_message, latest_query]
                new_messages = []

                token_count = sum([len(encoding.encode(m)) for m in keep_messages])
                # fit in as many of the previous human messages as possible
                for message in chat[1:-1:2]:
                    token_count += len(encoding.encode(message.text))

                    if token_count > int(os.environ['MAX_HUMAN_TOKENS']):
                        break

                    new_messages.append(message.text)
                    
                #query_messages = [system_message] + new_messages + [latest_query]
                #query_text = '\n'.join(query_messages)

                # add some more context
                docs = embedding_search(latest_query, k=int(os.environ['NUM_RELEVANT_FOLLOWUP_DOCS']))
                context = format_context(docs)
                formatted_query = format_query(latest_query, context)
            else:
                formatted_query = latest_query

            # always include the system message and the latest query in the prompt
            system_message_w_role = SystemMessage(content=system_message)
            latest_query_w_role = HumanMessage(content=formatted_query)
            messages = [latest_query_w_role]

            # for all the rest of the messages, iterate over them in reverse and fit as many in as possible
            token_limit = int(os.environ['MAX_TOKENS'])
            num_tokens = len(encoding.encode(system_message)) + len(encoding.encode(formatted_query))
            for message in reversed(chat[1:-1]):
                # count the number of new tokens
                num_tokens += int(os.environ['TOKENS_PER_MESSAGE'])
                num_tokens += len(encoding.encode(message.text))

                if num_tokens > token_limit:
                    # if we're over the token limit, stick with what we've got
                    break
                else:
                    # otherwise, add the new message in after the system prompt, but before the rest of the messages we've added
                    new_message = HumanMessage(content=message.text) if message.sender == 'user' else AIMessage(content=message.text)
                    messages = [new_message] + messages

            # add the system message to the beginning of the prompt
            messages = [system_message_w_role] + messages
            print(messages)

            llm(messages)

        finally:
            g.close()

    def chat_fn(prompt):
        g = ThreadedGenerator()
        threading.Thread(target=llm_thread, args=(g, prompt)).start()
        return g

    return StreamingResponse(chat_fn(chat), media_type='text/event-stream')
