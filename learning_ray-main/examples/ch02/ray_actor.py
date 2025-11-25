import ray
import time

database = [
    "Learning", "Ray",
    "Flexible", "Distributed", "Python", "for", "Machine", "Learning"
]

db_object_ref = ray.put(database)

@ray.remote
class DataTracker:
    def __init__(self):
        self._counts = 0

    def increment(self):
        self._counts += 1

    def counts(self):
        return self._counts

@ray.remote
def retrieve_tracker_task(item, tracker, db):
    time.sleep(item / 10.)
    tracker.increment.remote()
    return item, db[item]


tracker = DataTracker.remote()

object_references = [
    retrieve_tracker_task.remote(item, tracker, db_object_ref) for item in range(8)
]
data = ray.get(object_references)

print(data)
print(ray.get(tracker.counts.remote()))

