import uvicorn
from fastapi import FastAPI

from routers.cookies import router as cookies_router

app = FastAPI()

app.include_router(cookies_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
