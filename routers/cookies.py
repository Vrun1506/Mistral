import json

from fastapi import APIRouter, Request, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from supabase import create_client

from config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
from services.claude_fetcher.master import ClaudeFetcher
from store import create_user_object_with_convos

router = APIRouter()

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# Takes in the cookies.json file
@router.post("/get-cookies")
async def get_cookies(request: Request, file: UploadFile = File(...)):
    ## USER VALIDATION
    # @supabase/ssr stores session in sb-<ref>-auth-token cookie (may be chunked: .0, .1, …)
    cookie_name = "sb-koiaoajdcnxsarfpsfau-auth-token"
    raw = request.cookies.get(cookie_name, "")
    if not raw:
        # Reassemble chunked cookies
        chunks = []
        i = 0
        while True:
            chunk = request.cookies.get(f"{cookie_name}.{i}", "")
            if not chunk:
                break
            chunks.append(chunk)
            i += 1
        raw = "".join(chunks)

    if not raw:
        raise HTTPException(status_code=401, detail="Missing auth cookie")

    try:
        session = json.loads(raw)
        access_token = session["access_token"]
    except (json.JSONDecodeError, KeyError):
        raise HTTPException(status_code=401, detail="Malformed auth cookie")

    user_response = supabase.auth.get_user(access_token)
    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user_response.user.id
    ## END USER VALIDATION

    # Load file in
    try:
        contents = await file.read()
        cookies = json.loads(contents)
    finally:
        await file.close()

    async def event_generator():
        fetcher = ClaudeFetcher(cookies)
        count = await fetcher.get_all_conversations()
        yield f"data: {json.dumps({'type': 'info', 'message': f'Found {count} chats'})}\n\n"

        uuids = await fetcher.fetch_conversation_list(total=count)
        yield f"data: {json.dumps({'type': 'info', 'message': f'Fetched all {len(uuids)} conversation IDs'})}\n\n"

        user = create_user_object_with_convos(user_id)
        async for chunk in fetcher.fetch_all_conversation_details(uuids):
            if chunk.get("conversation"):
                c = chunk["conversation"]
                user.upsert_conversation(c["uuid"], c["name"], c["messages"])
                yield f"data: {json.dumps({'type': 'info', 'message': chunk['message']})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': chunk['message']})}\n\n"
        
        # Debugging temporarily printing out the object and what it looks like in memory 
        summary = {
            "user_id": user_id,
            "total_conversations": len(user.conversations),
            "conversations": [
                {"uuid": c.uuid, "name": c.name, "message_count": len(c.messages)}
                for c in user.conversations.values()
            ],
        }
        print(json.dumps(summary, indent=2))

        yield f"data: {json.dumps({'type': 'info', 'message': f'All conversations downloaded and stored ({len(user.conversations)} total)'})}\n\n"
        await fetcher.close()

    return StreamingResponse(event_generator(), media_type="text/event-stream")
