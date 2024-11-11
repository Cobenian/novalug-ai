# from llama_agents import (
#     AgentService,
#     ControlPlaneServer,
#     SimpleMessageQueue,
#     AgentOrchestrator,
# )
# from llama_index.core.agent import FunctionCallingAgentWorker
# from llama_index.core.tools import FunctionTool
# from llama_index.llms.anthropic import Anthropic
# import logging

# from llama_agents import LocalLauncher

# # turn on logging so we can see the system working
# logging.getLogger("llama_agents").setLevel(logging.INFO)

# # Set up the message queue and control plane
# message_queue = SimpleMessageQueue()
# control_plane = ControlPlaneServer(
#     message_queue=message_queue,
#     orchestrator=AgentOrchestrator(llm=Anthropic()),
# )


# # create a tool
# def get_the_secret_fact() -> str:
#     """Returns the secret fact."""
#     return "The secret fact is: A baby llama is called a 'Cria'."


# tool = FunctionTool.from_defaults(fn=get_the_secret_fact)

# # Define an agent
# worker = FunctionCallingAgentWorker.from_tools([tool], llm=Anthropic())
# agent = worker.as_agent()

# # Create an agent service
# agent_service = AgentService(
#     agent=agent,
#     message_queue=message_queue,
#     description="General purpose assistant",
#     service_name="assistant",
# )


# launcher = LocalLauncher(
#     [agent_service],
#     control_plane,
#     message_queue,
# )

# # Run a single query through the system
# result = launcher.launch_single("What's the secret fact?")
# print(result)
