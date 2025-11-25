from ray import serve
from ray.serve.handle import DeploymentHandle


@serve.deployment
class Hello:
    def __call__(self) -> str:
        return "Hello"


@serve.deployment
class World:
    def __call__(self) -> str:
        return " world!"


@serve.deployment
class Ingress:
    def __init__(self, hello_handle: DeploymentHandle, world_handle: DeploymentHandle):
        self._hello_handle = hello_handle
        self._world_handle = world_handle

    async def __call__(self) -> str:
        hello_response = self._hello_handle.remote()
        world_response = self._world_handle.remote()
        return (await hello_response) + (await world_response)


hello = Hello.bind()
world = World.bind()

# The deployments passed to the Ingress constructor are replaced with handles.
app = Ingress.bind(hello, world)

# Deploys Hello, World, and Ingress.
handle: DeploymentHandle = serve.run(app)

# `DeploymentHandle`s can also be used to call the ingress deployment of an application.
assert handle.remote().result() == "Hello world!"
