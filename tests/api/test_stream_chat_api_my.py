import asyncio
from datetime import datetime

from fastapi import FastAPI
from sse_starlette import EventSourceResponse, ServerSentEvent

app = FastAPI(debug=True)


async def echo_stream(message: str):
    t = 0
    while True:
        await asyncio.sleep(1)
        yield ServerSentEvent(
            event="echo",
            id=t,
            data={
                "message": message,
                "created_at": datetime.now().timestamp(),
            },
        )
        t += 1


@app.get("/sse")
async def sse(message: str) -> EventSourceResponse:
    return EventSourceResponse(content=echo_stream(message))

# if __name__ == '__main__':
#     # uvicorn.run()
