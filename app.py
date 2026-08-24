from pathlib import Path
import traceback

import uvicorn
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from pypdf import PdfReader
import io

from backend import run_career_agent, resume_career_agent

# Kept from the original project to allow the existing synchronous agent
# functions to call async MCP helpers inside FastAPI.
import nest_asyncio

nest_asyncio.apply()

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="CareerMate AI",
    description=(
        "LangGraph Multi-Agent Job Application Assistant with Supervisor, "
        "Guardrails, Human-in-the-Loop, and FastAPI Frontend"
    ),
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1)
    approved: bool
    feedback: str = ""


def _extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text).strip()


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/career")
async def career_assistant(
    message: str = Form(...),
    thread_id: str | None = Form(None),
    resume_file: UploadFile | None = File(None),
):
    try:
        user_message = message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        resume_text = ""
        if resume_file is not None:
            if resume_file.content_type != "application/pdf":
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": "Only PDF resumes are supported right now.",
                    },
                )

            file_bytes = await resume_file.read()
            resume_text = _extract_pdf_text(file_bytes)

        result = run_career_agent(
            user_input=user_message,
            resume_text=resume_text,
            thread_id=thread_id,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.post("/api/career/approve")
async def approve_application(request_data: ApprovalRequest):
    try:
        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        result = resume_career_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        print("APPROVAL ERROR:", exc)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(exc),
            },
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "CareerMate AI API is running",
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
        ],
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )