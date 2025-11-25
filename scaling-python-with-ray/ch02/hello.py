import ray

ray.init(num_cpus=4)

def hi():
    import os
    import socket
    return f"Running on {socket.gethostname()} in pid {os.getpid()}"

@ray.remote
def remote_hi():
    import os
    import socket
    return f"Running on {socket.gethostname()} in pid {os.getpid()}"

print(hi())

future = remote_hi.remote()
print(ray.get(future))
