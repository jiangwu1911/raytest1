import requests
from starlette.requests import Request

from ray import serve


@serve.deployment
class MostBasicIngress:
    async def __call__(self, request: Request) -> str:
        name = (await request.json())["name"]
        return f"Hello {name}!"


app = MostBasicIngress.bind()
serve.run(app)
assert (
    requests.get("http://127.0.0.1:8000/", json={"name": "Corey"}).text
    == "Hello Corey!"
)
