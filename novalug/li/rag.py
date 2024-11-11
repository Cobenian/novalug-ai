# from llama_index.core.tools import QueryEngineTool, ToolMetadata
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# # from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.anthropic import Anthropic

# # embed_model = OpenAIEmbedding(api_key="sk-...")
# query_llm = Anthropic(model="claude-3-haiku-20240307", api_key="sk-ant-...")

# # load data
# uber_docs = SimpleDirectoryReader(input_files=["./data/10k/uber_2021.pdf"]).load_data()
# # build index
# uber_index = VectorStoreIndex.from_documents(uber_docs, embed_model=embed_model)
# uber_engine = uber_index.as_query_engine(similarity_top_k=3, llm=query_llm)
# query_engine_tool = QueryEngineTool(
#     query_engine=uber_engine,
#     metadata=ToolMetadata(
#         name="uber_10k",
#         description=(
#             "Provides information about Uber financials for year 2021. "
#             "Use a detailed plain text question as input to the tool."
#         ),
#     ),
# )
