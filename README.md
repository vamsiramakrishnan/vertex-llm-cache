## Vertex-LLM-Cache

The `VertexLLMCache` class is a hybrid cache that uses both semantic and full-text search to store and retrieve data. It is configurable using Elasticsearch. 

### Installation
To install the required packages, run the following command:
```
pip install langchain elasticsearch google-cloud-aiplatform
```

### How to use 
```python
import pprint
from vertex_llm_cache import VertexLLMCache, VertexLLMCacheConfig

cache_config = VertexLLMCacheConfig(
    host="your-host-ip",
    port=your-port,
    username='elastic',
    password='your-password',
    index_name='your_index_name',
    cert_path='your_cert_path'
)
cache = VertexLLMCache(config=cache_config)

question = "Your Question"
answer = "LLM Answer"
cache.insert_into_cache(question, answer)

next_qn = "How far is"
l1_results = cache.l1_search(next_qn)
l2_results = cache.l2_search(next_qn)
l3_results = cache.l3_search(next_qn)

print(f"L1 Results\n:{l1_results}")
print(f"L2 Results\n:{l2_results}")
print(f"L3 Results\n:{l3_results}")
```

### Documentation

#### How to use

To use the `VertexLLMCache` class, you must first create a `VertexLLMCacheConfig` object with the appropriate parameters. Then, you can create a `VertexLLMCache` object with the `VertexLLMCacheConfig` object as a parameter. 

Once you have created a `VertexLLMCache` object, you can insert data into the cache using the `insert_into_cache` method. This method takes two parameters: a question and an answer. The question is the key that will be used to retrieve the answer later. 

To retrieve data from the cache, you can use the `l1_search`, `l2_search`, and `l3_search` methods. These methods take a single parameter: a query string. The `l1_search` method searches for exact matches to the query string. The `l2_search` method searches for matches to the query string using semantic search. The `l3_search` method searches for matches to the query string using full-text search. 

#### Example

```python
import pprint
from vertex_llm_cache import VertexLLMCache, VertexLLMCacheConfig

cache_config = VertexLLMCacheConfig(
    host="your-host-ip",
    port=your-port,
    username='your-username',
    password='your-password',
    index_name='your-index-name',
    cert_path='your-cert-path'
)
cache = VertexLLMCache(config=cache_config)

question = "Your Question"
answer = "LLM Answer"
cache.insert_into_cache(question, answer)

next_qn = "How far is"
l1_results = cache.l1_search(next_qn)
l2_results = cache.l2_search(next_qn)
l3_results = cache.l3_search(next_qn)

print(f"L1 Results\n:{l1_results}")
print(f"L2 Results\n:{l2_results}")
print(f"L3 Results\n:{l3_results}")
```

