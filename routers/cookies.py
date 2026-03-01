import asyncio
import json
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from auth import get_current_user_id
from services.claude_fetcher.master import ClaudeFetcher
from services.pipeline.pipeline import run_pipeline_async
from store import create_user_object_with_convos


class FetchRequest(BaseModel):
    session_key: str
    last_active_org: str


router = APIRouter()


@router.post("/count-conversations")
async def count_conversations(body: FetchRequest, user_id: str = Depends(get_current_user_id)) -> dict[str, int]:
    """Return the visible conversation count (matches claude.ai web UI)."""
    fetcher = ClaudeFetcher(session_key=body.session_key, org_id=body.last_active_org)
    try:
        count = await fetcher.get_all_conversations()
        return {"count": count}
    finally:
        await fetcher.close()


@router.post("/get-cookies")
async def get_cookies(body: FetchRequest, user_id: str = Depends(get_current_user_id)) -> StreamingResponse:
    async def event_generator() -> AsyncGenerator[str, None]:
        fetcher = ClaudeFetcher(session_key=body.session_key, org_id=body.last_active_org)
        count = await fetcher.get_all_conversations()
        yield f"data: {json.dumps({'type': 'info', 'message': f'Found {count} chats'})}\n\n"

        uuids = await fetcher.fetch_conversation_list(limit=count)
        yield f"data: {json.dumps({'type': 'info', 'message': f'Fetched all {len(uuids)} conversation IDs'})}\n\n"

        user = create_user_object_with_convos(user_id)
        async for chunk in fetcher.fetch_all_conversation_details(uuids):
            if chunk.get("conversation"):
                c = chunk["conversation"]
                user.upsert_conversation(c["uuid"], c["name"], c["messages"])
                yield f"data: {json.dumps({'type': 'info', 'message': chunk['message']})}\n\n"
            else:
                yield f"data: {json.dumps({'type': 'error', 'message': chunk['message']})}\n\n"

        await fetcher.close()

        done_msg = f"All conversations downloaded and stored ({len(user.conversations)} total)"
        yield f"data: {json.dumps({'type': 'info', 'message': done_msg})}\n\n"

        # --- Run pipeline on the stored conversations ---
        # Convert to match pipeline input from the object we have in storage
        conversations = user.as_pipeline_input()
        if not conversations:
            yield f"data: {json.dumps({'type': 'error', 'message': 'No conversations to process'})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'info', 'message': 'Starting pipeline...'})}\n\n"

        progress_queue: asyncio.Queue[str | None] = asyncio.Queue()

        def on_progress(phase: object, message: str, progress: float, **_kwargs: object) -> None:
            event = {"type": "pipeline", "phase": str(phase), "message": message, "progress": progress}
            progress_queue.put_nowait(json.dumps(event))

        async def _run() -> None:
            try:
                topic_groups, hierarchy = await run_pipeline_async(
                    conversations=conversations,  # type: ignore[arg-type]
                    on_progress=on_progress,
                )
                user.set_pipeline_results(topic_groups, hierarchy)
            except Exception as e:
                progress_queue.put_nowait(json.dumps({"type": "error", "message": f"Pipeline error: {e}"}))
            finally:
                progress_queue.put_nowait(None)

        task = asyncio.create_task(_run())

        while True:
            event_data = await progress_queue.get()
            if event_data is None:
                break
            yield f"data: {event_data}\n\n"

        await task

        n_topics = len(user.topic_groups or {})
        msg = f"Pipeline complete — {n_topics} topics, hierarchy ready"
        yield f"data: {json.dumps({'type': 'info', 'message': msg})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
