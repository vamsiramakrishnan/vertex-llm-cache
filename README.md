# vertex-llm-cache
A Hybrid (Semantic + Full-Text) cache configurable using elasticsearch 

How to use 
```
import pprint
cache_config = VertexLLMCacheConfig(host="10.128.15.232", port=9200, username='elastic', password='OMAridSl6MRMWAGDn*6z', index_name='vamramak_llm_cache', cert_path = './instance_ss.crt')
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
