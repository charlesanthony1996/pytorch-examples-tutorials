import aiohttp
import asyncio

api_url = "https://jsonplaceholder.typicode.com/users"

async def fetch_user(session, user_id):
    # fetch a user by id
    async with session.get(f"{api_url}/{user_id}") as response:
        return await response.json()


async def batch_fetch_users(user_ids):
    # fetch mutiple users in batch
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_user(session, user_id) for user_id in user_ids]
        return await asyncio.gather(*tasks)

def main():
    user_ids_batch = [1, 2, 3, 4, 5]
    users = asyncio.run(batch_fetch_users(user_ids_batch))
    for user in users:
        print(user["name"])


if __name__ == "__main__":
    main()