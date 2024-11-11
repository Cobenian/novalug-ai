import weaviate
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
import os
from termcolor import cprint

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with weaviate.connect_to_local() as client:

    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name="GameRecapIdx", text_key="content"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    # retrieve matching nodes
    retriever = index.as_retriever(similarity_top_k=1)
    nodes = retriever.retrieve("Who was the starting pitcher?")
    cprint("matching nodes:", "blue")
    pprint_source_node(nodes[0])
    print("")
    print("")

    # answer questions
    query_engine = index.as_query_engine(llm=Anthropic())
    response = query_engine.query("Which teams did the Thundercats play against?")
    cprint("our opponents", "blue")
    print(response)
    print("")
    response = query_engine.query(
        "Who was the starting pitcher for the Thundercats against the Braddock Bearcats?"
    )
    cprint("our starting pitcher", "blue")
    print(response)
    print("")
    response = query_engine.query(
        "Who had the most hits on the Thundercats vs the Bearcats?"
    )
    cprint("our leading hitter", "blue")
    print(response)
    print("")
    response = query_engine.query("Who was the starting pitcher for the Diamond Dogs?")
    cprint("oppsosing starting pitcher?", "blue")
    print(response)
    print("")
