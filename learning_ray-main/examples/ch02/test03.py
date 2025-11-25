import time
import ray

database = [
    "Learning", "Ray",
    "Flexible", "Distributed", "Python", "for", "Machine", "Learning"
]

db_object_ref = ray.put(database)

def print_runtime(input_data, start_time):
    print(f'Runtime: {time.time() - start_time:.2f} seconds, data:')
    print(*input_data, sep="\n")

def retrieve(item):
    time.sleep(item / 10.)
    return item, database[item]

@ray.remote
def retrieve_task(item, db):
    time.sleep(item / 10.)
    return item, db[item]

@ray.remote
def follow_up_task(retrieve_result):
    original_item, _ = retrieve_result
    follow_up_result = retrieve(original_item + 1)
    return retrieve_result, follow_up_result

retrieve_refs = [retrieve_task.remote(item, db_object_ref) for item in [0, 2, 4, 6]]
follow_up_refs = [follow_up_task.remote(ref) for ref in retrieve_refs]

result = [print(data) for data in ray.get(follow_up_refs)]
