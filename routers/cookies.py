from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class CookiesRequest(BaseModel):
    cookies: dict[str, str]


@router.post("/get-cookies")
async def get_cookies(body: CookiesRequest):
    print("Cookies Received")
    return {"message": "Cookies Received"}
