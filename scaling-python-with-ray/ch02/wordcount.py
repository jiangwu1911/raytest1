import ray

ray.init(num_cpus=4)

urls = ray.data.from_items([
    "https://github.com/scalingpythonml/scalingpythonml",
    "https://github.com/ray-project/ray"])

def fetch_page(url):
    import requests
    f = requests.get(url)
    return f.text

pages = urls.map(fetch_page)
pages.take(1)

words = pages.flat_map(lambda x: x.split(" ")).map(lambda w: (w, 1))
grouped_words = words.groupby(lambda wc: wc[0])
word_counts = grouped_words.count()
word_counts.show()
