import weaviate
from llama_index.core import SimpleDirectoryReader
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SimpleFileNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
from sentence_transformers import SentenceTransformer
from termcolor import cprint
import os

# Set the environment variable to disable parallelism in tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_season_chat():

    with weaviate.connect_to_local() as client:

        # load the blogs in using the reader
        game_recaps = SimpleDirectoryReader("./data/game_recaps").load_data()

        # Chunk up the game recaps into nodes
        parser = SimpleFileNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(game_recaps)

        # Initialize the Sentence Transformer model
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Generate embeddings for each node
        for node in nodes:
            node.embedding = model.encode(node.text)

        # construct vector store
        vector_store = WeaviateVectorStore(
            weaviate_client=client,
            index_name="GameRecapIdx",
            text_key="content",
        )

        # setting up the storage for the embeddings
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

        # set up the index
        index = VectorStoreIndex(
            nodes, storage_context=storage_context, embed_model=embed_model
        )

        query_engine = index.as_query_engine(llm=Anthropic())
        q = "Which teams did the Thundercats play against?"
        cprint(f"Q: {q}", "blue")
        response = query_engine.query(q)
        cprint(response, "yellow")
        print("")

        chat_engine = index.as_chat_engine(llm=Anthropic())

        q = "Who had the most hits for the Thundercats in each game?"
        cprint(f"Q: {q}", "blue")
        response = chat_engine.chat(q)
        cprint(response, "yellow")
        print("")

        q = "Who the Thundercats play next?"
        cprint(f"Q: {q}", "blue")
        response = chat_engine.chat(q)
        cprint(response, "yellow")
        print("")

        q = "When do they play them?"
        cprint(f"Q: {q}", "blue")
        response = chat_engine.chat(q)
        cprint(response, "yellow")
        print("")


def main():
    run_season_chat()
