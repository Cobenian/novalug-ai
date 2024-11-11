import weaviate
import json
from weaviate.classes.query import MetadataQuery


from llama_index.vector_stores.weaviate import WeaviateVectorStore

# from llama_index.core import VectorStoreIndex
from llama_index.core.response.pprint_utils import pprint_source_node

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext

from llama_index.llms.anthropic import Anthropic

with weaviate.connect_to_local() as client:

    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name="GameRecapIdx", text_key="content"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    index = VectorStoreIndex.from_vector_store(
        vector_store, llm=None, embed_model=embed_model
    ).as_retriever(
        similarity_top_k=1, embed_model=embed_model, storage_context=storage_context
    )

    # query_engine = index.as_query_engine(llm=Anthropic())
    # response = query_engine.query("Which teams did the Thundercats play against?")
    # print(response)

    nodes = index.retrieve("Who was the starting pitcher?")

    pprint_source_node(nodes[0])

# with weaviate.connect_to_local() as client:
#     game_recaps = client.collections.get("GameRecapIdx")


# response = game_recaps.query.fetch_objects()

# for o in response.objects:
#     print(o.properties)


# response = game_recaps.query.near_text(
#     query="starting pitcher",
#     limit=2,
#     return_metadata=MetadataQuery(distance=True)
# )

# for o in response.objects:
#     print(o.properties)
#     print(o.metadata.distance)


# from llama_index.core.retrievers import VectorIndexAutoRetriever
# from llama_index.core.vector_stores.types import MetadataInfo, VectorStoreInfo

# from llama_index.core import SimpleDirectoryReader
# from llama_index.vector_stores.weaviate import WeaviateVectorStore
# from llama_index.core import VectorStoreIndex, StorageContext
# from llama_index.core.node_parser import SimpleFileNodeParser
# from sentence_transformers import SentenceTransformer
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.anthropic import Anthropic
# import os

# # Set the environment variable to disable parallelism in tokenizers
# os.environ["TOKENIZERS_PARALLELISM"] = "false"


# client = weaviate.connect_to_local()

# # load the blogs in using the reader
# game_recaps = SimpleDirectoryReader("./data/game_recaps").load_data()


# # Chunk up the game recaps into nodes
# parser = SimpleFileNodeParser.from_defaults()
# nodes = parser.get_nodes_from_documents(game_recaps)

# # Initialize the Sentence Transformer model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Generate embeddings for each node
# for node in nodes:
#     node.embedding = model.encode(node.text)


# # chunk up the blog posts into nodes
# # parser = SimpleFileNodeParser.from_defaults(chunk_size=1024, chunk_overlap=20)
# # nodes = parser.get_nodes_from_documents(game_recaps)

# # construct vector store
# vector_store = WeaviateVectorStore(
#     weaviate_client=client, index_name="GameRecapsIdx"
# )

# # setting up the storage for the embeddings
# storage_context = StorageContext.from_defaults(vector_store=vector_store)


# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# # set up the index
# index = VectorStoreIndex(
#     nodes, storage_context=storage_context, embed_model=embed_model
# )

# vector_store_info = VectorStoreInfo(
#     content_info="recaps of baseball games",
#     metadata_info=[
#         MetadataInfo(
#             name="content",
#             type="str",
#             description=(
#                 "Summary of the baseball game, including the teams that played,"
#                 "the pitchers, batters and key plays."
#             ),
#         ),
#     ],
# )

# retriever = VectorIndexAutoRetriever(
#     index, vector_store_info=vector_store_info, llm=Anthropic()
# )

# response = retriever.retrieve("Who was the starting pitcher?")
# print(response[0])

# game_recaps = client.collections.get("GameRecapIdx")

# questions = client.collections.get("Question")

# response = game_recaps.query.near_text(
#     query="Dylan",
#     limit=2
# )

# for obj in response.objects:
#     print(json.dumps(obj.properties, indent=2))

# client.close()  # Free up resources
