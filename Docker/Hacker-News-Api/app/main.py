import json
from fastapi import FastAPI, Query
import httpx
from redis import asyncio as aioredis
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

app = FastAPI()

HACKER_NEWS_API_BASE_URL = "https://hacker-news.firebaseio.com/v0"

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost", encoding="utf8", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")

def custom_key_builder(func, *args, **kwargs):
    count = kwargs.get('count', 10)
    return f"top_stories:{count}"

@app.get("/top-stories")
@cache(expire=300, key_builder=custom_key_builder)
async def get_top_stories(count: int = Query(default=10, ge=1, le=100)):
    async with httpx.AsyncClient() as client:
        # Fetch top story IDs
        response = await client.get(f"{HACKER_NEWS_API_BASE_URL}/topstories.json")
        response.raise_for_status()
        all_story_ids = response.json()
        story_ids = all_story_ids[:count]  # Get the specified number of stories

        # Fetch details for each story
        stories = []
        for story_id in story_ids:
            story_response = await client.get(f"{HACKER_NEWS_API_BASE_URL}/item/{story_id}.json")
            story_response.raise_for_status()
            story = story_response.json()
            stories.append({
                "id": story["id"],
                "title": story["title"],
                "text": story.get("text", ""),  # Use .get() to avoid KeyError if "text" is missing
                "score": story["score"],
                "by": story["by"]
            })

    return {"top_stories": stories, "count": len(stories)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)