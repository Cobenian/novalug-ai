import weaviate

# this exists to make sure the Weaviate server is running (in a docker container)

client = weaviate.connect_to_local()

print(client.is_ready())  # Should print: `True`

client.close()
