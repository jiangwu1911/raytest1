import ray

ray.init(num_cpus=4)

@ray.remote
class HelloWorld(object):
    def __init__(self):
        self.value = 0

    def greet(self):
        self.value += 1
        return f"Hi user #{self.value}"

hello_actor = HelloWorld.remote()
print(ray.get(hello_actor.greet.remote()))
