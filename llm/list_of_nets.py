import base64
import ollama
from httpx import DigestAuth

client = ollama.Client(
            host="https://ollama.kky.zcu.cz",
            auth=DigestAuth(
                "username", 
                "password"
            ),
        )

response = client.list()

for model in response.models:
    print(model.model)