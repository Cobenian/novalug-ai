import weaviate

# Connect to Weaviate
with weaviate.connect_to_local() as client:

    # Define the schema for the GameRecapIdx collection
    schema = {
        "class": "GameRecapIdx",
        "description": "A collection of game recaps",
        "vectorizer": "text2vec-transformers",  # Specify the vectorizer module
        "properties": [
            {
                "name": "title",
                "dataType": ["text"],
            },
            {
                "name": "content",
                "dataType": ["text"],
            },
        ],
    }

    # Create the schema
    client.schema.create_class(schema)
