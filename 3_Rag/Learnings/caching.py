import redis
import json

cache = redis.Redis(host = 'localhost',port = 6379,db=0)

def get_cached_response(query):
    cached_result = cache.get(query)
    return json.loads(cached_result) if cached_cached else None

def store_response_in_cache(query,response):
    cache.set(query,json.dumps(response),ex=3600)

query = "what is cricket?"
cached_response = get_cached_response(query)

if cached_response:
    print(f"cached response is {cached_response}")
else:
    response =qa_chain.run(query)
    store_response_in_cache(query,response)
    print("the response for the qsn is {response}")