import weaviate
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.weaviate import WeaviateVectorStore
from llama_index.core.response.pprint_utils import pprint_source_node
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.anthropic import Anthropic
import os
from termcolor import cprint
from llama_index.core.tools import QueryEngineTool, FunctionTool
import nest_asyncio
import pandas as pd


from llama_index.core.agent import (
    StructuredPlannerAgent,
    FunctionCallingAgentWorker,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def scores_data():
    filename = "data/schedule/schedule_with_scores.csv"
    # read in the file as a pandas dataframe
    df = pd.read_csv(filename)
    return df


with weaviate.connect_to_local() as client:

    vector_store = WeaviateVectorStore(
        weaviate_client=client, index_name="GameRecapIdx", text_key="content"
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

    llm_anthropic = Anthropic(model="claude-3-opus-20240229")

    # answer questions
    query_engine = index.as_query_engine(llm=llm_anthropic)

    game_recap_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="game-recaps",
        description="Useful for asking questions about the Thundercats baseball games.",
    )

    scores_tool = FunctionTool.from_defaults(
        fn=scores_data,
        name="scores",
        description="Get the scores of the Thundercats games.",
    )

    # create the function calling worker for reasoning
    worker = FunctionCallingAgentWorker.from_tools(
        [game_recap_tool, scores_tool], verbose=True, llm=llm_anthropic
    )

    # wrap the worker in the top-level planner
    agent = StructuredPlannerAgent(
        worker, tools=[game_recap_tool], verbose=True, llm=llm_anthropic
    )

    nest_asyncio.apply()

    response = agent.chat("What was the closest game the Thundercats played?")

    print(response)
