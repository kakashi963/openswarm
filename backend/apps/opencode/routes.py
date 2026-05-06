from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.apps.opencode.managed import get_or_create_opencode_server, stop_opencode_server

router = APIRouter(prefix="/api/opencode", tags=["opencode"])


class StartRequest(BaseModel):
    workspace_path: str
    workspace_id: str


class StartResponse(BaseModel):
    url: str
    username: str
    password: str
    port: int
    message: str = "OpenCode server started successfully"


@router.post("/start", response_model=StartResponse)
async def start_opencode(req: StartRequest):
    """Start (or return existing) managed OpenCode server for a workspace.

    Frontend can then use @opencode-ai/sdk/v2/client with the returned URL + basic auth.
    """
    try:
        details = await get_or_create_opencode_server(
            workspace_path=req.workspace_path,
            workspace_id=req.workspace_id,
        )
        return StartResponse(
            url=details["url"],
            username=details["username"],
            password=details["password"],
            port=details["port"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stop/{workspace_id}")
async def stop_opencode(workspace_id: str):
    stop_opencode_server(workspace_id)
    return {"status": "stopped", "workspace_id": workspace_id}
