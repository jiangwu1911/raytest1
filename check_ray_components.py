import pkg_resources

def check_all_components():
    print("Checking all Ray components")
    
    components = [
        ("ray", "core"),
        ("ray.tune", "tune"),
        ("ray.rllib", "rllib"),
        ("ray.train", "train"),
        ("ray.serve", "serve"),
        ("ray.data", "data"),
        ("ray.job_submission", "job_submission"),
        ("ray.dashboard", "dashboard"),
        ("ray.autoscaler", "autoscaler"),
        ("ray._private.ray_constants", "internal"),
    ]
    
    for module, description in components:
        try:
            __import__(module)
            print(f"OK - {description:15} ({module})")
        except ImportError as e:
            print(f"FAIL - {description:15} ({module})")

if __name__ == "__main__":
    check_all_components()
