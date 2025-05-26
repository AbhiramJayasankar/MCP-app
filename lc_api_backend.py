import os
import time
import uuid
from typing import List, Optional, Dict, Any, AsyncGenerator
from datetime import datetime, timezone
from urllib.parse import urlparse
import asyncio
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from dotenv import load_dotenv

# Langchain components
try:
    from langchain_ollama import ChatOllama
except ImportError:
    print("langchain-ollama package not found. Please install with 'pip install langchain-ollama'")
    print("Falling back to deprecated langchain_community.chat_models for ChatOllama.")
    from langchain_community.chat_models import ChatOllama

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

import ollama

load_dotenv()

# --- Pydantic Models (Keep as is from previous version) ---
class ChatMessageInput(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    model_id: Optional[str] = None
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    user: Optional[str] = None
    n: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = 0
    presence_penalty: Optional[float] = 0
    stop: Optional[List[str]] = None
    logit_bias: Optional[Dict[str, float]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    @field_validator('messages')
    @classmethod
    def check_messages_not_empty(cls, v):
        if not v: raise ValueError("messages list cannot be empty")
        return v

class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
class ChatCompletionChunkChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None
class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChunkChoice]
class ChatMessageOutput(BaseModel):
    role: str
    content: str
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessageOutput
    finish_reason: Optional[str] = "stop"
class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
class ModelPermission(BaseModel):
    id: str = Field(default_factory=lambda: f"modelperm-{uuid.uuid4().hex}")
    object: str = "model_permission"
    created: int = Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: Optional[str] = None
    is_blocking: bool = False
class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "ollama"
    permission: List[ModelPermission] = Field(default_factory=list)
    root: Optional[str] = None
    parent: Optional[str] = None
class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

# --- FastAPI Application Setup (Keep as is) ---
app = FastAPI(
    title="Langchain Ollama OpenAI-Compatible API with Corrected Model Listing",
    description="An API endpoint to serve Langchain Ollama models and a dummy model, compatible with OpenWebUI.",
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "llama3")
OPENWEBUI_ORIGIN = os.getenv("OPENWEBUI_ORIGIN", "http://localhost:3000")
DUMMY_MODEL_ID = "dummy-model-from-fastapi-v1"
app.add_middleware(CORSMiddleware, allow_origins=[OPENWEBUI_ORIGIN, "http://localhost:8080", "http://127.0.0.1:3000", "http://127.0.0.1:8080", "http://localhost:5173", "http://127.0.0.1:5173"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.middleware("http")
async def debug_request_body(request: Request, call_next): # (Keep as is)
    if "/v1/chat/completions" in request.url.path and request.method == "POST":
        body_bytes = await request.body()
        try:
            body_json = json.loads(body_bytes.decode('utf-8'))
            print("--- Incoming /v1/chat/completions Request Body ---"); print(json.dumps(body_json, indent=2)); print("----------------------------------------------------")
        except Exception as e:
            print(f"Error decoding chat/completions request body as JSON: {e}"); print(f"Raw body: {body_bytes[:500]}...")
        request._body = body_bytes
    response = await call_next(request)
    return response

@app.exception_handler(RequestValidationError) # (Keep as is)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("--- Request Validation Error ---"); print(f"Path: {request.url.path}"); print(f"Method: {request.method}")
    try:
        body_for_error_log = await request.json(); print("Request Body (if JSON parseable at error time):"); print(json.dumps(body_for_error_log, indent=2))
    except Exception: print("Request Body: (See middleware log for /v1/chat/completions POST requests)")
    print("Validation Errors (from Pydantic):"); print(json.dumps(exc.errors(), indent=2)); print("------------------------------")
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

llm = None; ollama_client = None # (Keep Langchain/Ollama client init as is)
try:
    llm = ChatOllama(model=OLLAMA_DEFAULT_MODEL, temperature=0.7, base_url=OLLAMA_BASE_URL)
    print(f"Successfully initialized default ChatOllama instance with model: {OLLAMA_DEFAULT_MODEL} at {OLLAMA_BASE_URL}")
    parsed_url = urlparse(OLLAMA_BASE_URL); ollama_client_host = f"{parsed_url.scheme}://{parsed_url.netloc}"
    ollama_client = ollama.Client(host=ollama_client_host)
    print(f"Ollama client for listing models initialized for host: {ollama_client_host}")
except Exception as e: print(f"Error initializing default ChatOllama LLM or Ollama client: {e}.")

prompt_template = ChatPromptTemplate.from_messages([MessagesPlaceholder(variable_name="history"), ("user", "{input}")])
def convert_messages_to_langchain_format(messages: List[ChatMessageInput]) -> List[BaseMessage]: # (Keep as is)
    lc_messages = []; 
    for msg_input in messages:
        if msg_input.role == "user": lc_messages.append(HumanMessage(content=msg_input.content))
        elif msg_input.role == "assistant": lc_messages.append(AIMessage(content=msg_input.content))
        elif msg_input.role == "system": lc_messages.append(SystemMessage(content=msg_input.content))
    return lc_messages
# --- API Endpoints ---
@app.get("/v1/models", response_model=ModelList)
async def list_models_endpoint():
    print("--- Attempting to list models (/v1/models endpoint hit) ---")
    model_cards = []
    if ollama_client:
        print("Ollama client IS initialized. Attempting to fetch models from Ollama.")
        try:
            ollama_list_response_obj = await run_in_threadpool(ollama_client.list) # This is ollama._types.ListResponse
            print(f"Raw response type from ollama_client.list(): {type(ollama_list_response_obj)}")
            # print(f"Raw response value from ollama_client.list(): {ollama_list_response_obj}") # Can be very verbose

            actual_models_list_from_ollama = []
            # The ollama.Client().list() returns a dictionary-like object (TypedDict)
            # which has a 'models' key. The value of 'models' is a list of
            # dictionary-like objects (ModelResponse TypedDicts).
            if isinstance(ollama_list_response_obj, dict) and 'models' in ollama_list_response_obj:
                actual_models_list_from_ollama = ollama_list_response_obj['models']
                print(f"Extracted 'models' list from ListResponse dict. Count: {len(actual_models_list_from_ollama)}")
            # Some versions or wrappers might return an object with a .models attribute
            elif hasattr(ollama_list_response_obj, 'models') and isinstance(getattr(ollama_list_response_obj, 'models'), list):
                actual_models_list_from_ollama = getattr(ollama_list_response_obj, 'models')
                print(f"Accessed .models attribute. Count: {len(actual_models_list_from_ollama)}")
            else:
                print(f"Could not extract models list from Ollama response. Type: {type(ollama_list_response_obj)}")

            if isinstance(actual_models_list_from_ollama, list):
                print(f"Processing models list. Number of models from Ollama: {len(actual_models_list_from_ollama)}")
                if not actual_models_list_from_ollama:
                    print("Ollama returned an empty list of models.")
                
                for model_data_item in actual_models_list_from_ollama: # model_data_item is a dict (ModelResponse TypedDict)
                    model_id_from_ollama = model_data_item.get("model") # Changed from 'name' to 'model'
                    modified_at_dt_obj = model_data_item.get("modified_at") 
                    
                    print(f"Processing model from Ollama: {model_id_from_ollama}")
                    if not model_id_from_ollama:
                        print(f"Skipping a model entry from Ollama due to missing 'model' key. Entry: {model_data_item}")
                        continue
                    
                    created_timestamp = int(time.time()) # Default
                    if modified_at_dt_obj and isinstance(modified_at_dt_obj, datetime):
                        try:
                            if modified_at_dt_obj.tzinfo is None:
                                modified_at_dt_obj = modified_at_dt_obj.replace(tzinfo=timezone.utc)
                            created_timestamp = int(modified_at_dt_obj.timestamp())
                        except Exception as ts_e:
                            print(f"Warning: Could not process datetime object for model {model_id_from_ollama}: '{modified_at_dt_obj}'. Error: {ts_e}")
                    elif modified_at_dt_obj:
                         print(f"Warning: 'modified_at' for model {model_id_from_ollama} is not a datetime object: {type(modified_at_dt_obj)}")

                    model_cards.append(ModelCard(id=model_id_from_ollama, created=created_timestamp, owned_by="ollama", root=model_id_from_ollama, permission=[ModelPermission()]))
            else:
                print(f"After attempting extraction, actual_models_list_from_ollama is not a list. Type: {type(actual_models_list_from_ollama)}")

        except Exception as e:
            print(f"CRITICAL ERROR fetching or processing models from Ollama: {e}. Ollama models will not be listed.")
    else:
        print("CRITICAL WARNING: Ollama client not initialized. Real Ollama models cannot be listed.")

    print(f"Adding dummy model: {DUMMY_MODEL_ID}")
    dummy_model_card = ModelCard(id=DUMMY_MODEL_ID, created=int(time.time()), owned_by="fastapi-custom-backend", root=DUMMY_MODEL_ID, permission=[ModelPermission()])
    model_cards.append(dummy_model_card)
    
    print(f"Final list of model cards to be returned (count: {len(model_cards)}): {[mc.id for mc in model_cards]}")
    return ModelList(data=model_cards)

# stream_generator and chat_completions endpoints (Keep as is from previous version)
async def stream_generator(actual_model_identifier: str, langchain_messages: List[BaseMessage], request_temperature: Optional[float], request_max_tokens: Optional[int]) -> AsyncGenerator[str, None]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"; created_time = int(time.time())
    if actual_model_identifier == DUMMY_MODEL_ID:
        yield f"data: {ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role='assistant'))]).model_dump_json()}\n\n"
        last_user_message_content = "nothing specific"
        if langchain_messages and isinstance(langchain_messages[-1], HumanMessage): last_user_message_content = langchain_messages[-1].content
        dummy_response_text = f"Hello from your FastAPI backend! This is the '{DUMMY_MODEL_ID}'. You said: '{last_user_message_content}'."; words = dummy_response_text.split(" ")
        for i, word in enumerate(words):
            chunk_content = word + (" " if i < len(words) - 1 else ""); stream_chunk = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=chunk_content))])
            yield f"data: {stream_chunk.model_dump_json()}\n\n"; await asyncio.sleep(0.05)
        final_chunk = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")])
        yield f"data: {final_chunk.model_dump_json()}\n\n"; yield "data: [DONE]\n\n"; return
    if llm is None:
        error_content = "[LLM Error: Default LLM not initialized. Cannot process request.]"
        yield f"data: {ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role='assistant'))]).model_dump_json()}\n\n"
        error_chunk_resp = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=error_content), finish_reason="error")])
        yield f"data: {error_chunk_resp.model_dump_json()}\n\n"; yield f"data: {ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason='error')]).model_dump_json()}\n\n"; yield "data: [DONE]\n\n"; return
    yield f"data: {ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(role='assistant'))]).model_dump_json()}\n\n"
    llm_params: Dict[str, Any] = {"model": actual_model_identifier, "base_url": OLLAMA_BASE_URL}; default_temp = 0.7
    if hasattr(llm, 'temperature') and llm.temperature is not None: default_temp = llm.temperature
    llm_params["temperature"] = request_temperature if request_temperature is not None else default_temp
    if request_max_tokens is not None and request_max_tokens > 0: llm_params["num_predict"] = request_max_tokens
    try: current_llm = ChatOllama(**llm_params)
    except Exception as e:
        error_content = f"[LLM Init Error for model {actual_model_identifier}: {str(e)}]"; error_chunk = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=error_content), finish_reason="error")])
        yield f"data: {error_chunk.model_dump_json()}\n\n"; yield "data: [DONE]\n\n"; print(f"Error initializing ChatOllama for stream: {e}"); return
    current_chain = prompt_template | current_llm | StrOutputParser(); history_messages = langchain_messages[:-1]; user_input_message = langchain_messages[-1]
    if not isinstance(user_input_message, HumanMessage):
        error_content = "[Error: Last message must be from user for this chain setup]"; error_chunk = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=error_content), finish_reason="stop")])
        yield f"data: {error_chunk.model_dump_json()}\n\n"; yield "data: [DONE]\n\n"; return
    input_data = {"history": history_messages, "input": user_input_message.content}
    try:
        async for chunk_content in current_chain.astream(input_data):
            stream_chunk = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=chunk_content), finish_reason=None)])
            yield f"data: {stream_chunk.model_dump_json()}\n\n"
    except Exception as e:
        error_content = f"[LLM Stream Error: {str(e)}]"; error_chunk = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(content=error_content), finish_reason="error")])
        yield f"data: {error_chunk.model_dump_json()}\n\n"; print(f"Error during LLM stream: {e}")
    final_chunk = ChatCompletionStreamResponse(id=completion_id, model=actual_model_identifier, created=created_time, choices=[ChatCompletionChunkChoice(index=0, delta=DeltaMessage(), finish_reason="stop")])
    yield f"data: {final_chunk.model_dump_json()}\n\n"; yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions", response_model=None) # (Keep as is)
async def chat_completions(request_data: ChatCompletionRequest, http_request_obj: Request):
    actual_model_identifier = request_data.model_id if request_data.model_id is not None else request_data.model
    if actual_model_identifier is None: raise HTTPException(status_code=400, detail="Field 'model' or 'model_id' is required in the request body.")
    if actual_model_identifier == DUMMY_MODEL_ID:
        last_user_message_content = "nothing specific"
        if request_data.messages:
            for msg_input in reversed(request_data.messages):
                if msg_input.role == "user": last_user_message_content = msg_input.content; break
        dummy_content = f"Hello from your FastAPI backend! This is the non-streamed '{DUMMY_MODEL_ID}'. You said: '{last_user_message_content}'."; response_message = ChatMessageOutput(role="assistant", content=dummy_content)
        choice = ChatCompletionChoice(index=0, message=response_message, finish_reason="stop"); prompt_tokens_dummy = sum(len(msg.content.split()) for msg in request_data.messages); completion_tokens_dummy = len(dummy_content.split())
        usage_stats = Usage(prompt_tokens=prompt_tokens_dummy, completion_tokens=completion_tokens_dummy, total_tokens=prompt_tokens_dummy + completion_tokens_dummy)
        return ChatCompletionResponse(model=DUMMY_MODEL_ID, choices=[choice], usage=usage_stats)
    if llm is None: raise HTTPException(status_code=503, detail=f"Default Ollama LLM ({OLLAMA_DEFAULT_MODEL}) not initialized.")
    if not request_data.messages: raise HTTPException(status_code=400, detail="No messages provided.")
    langchain_messages = convert_messages_to_langchain_format(request_data.messages)
    if not langchain_messages or not isinstance(langchain_messages[-1], (HumanMessage, SystemMessage)):
        is_last_message_user = isinstance(langchain_messages[-1], HumanMessage)
        if not is_last_message_user and any(isinstance(m, HumanMessage) for m in langchain_messages): print(f"Warning: Last message is {type(langchain_messages[-1])}, not HumanMessage.")
        elif not is_last_message_user: raise HTTPException(status_code=400, detail="Last message must be 'user'.")
    if request_data.stream:
        return StreamingResponse(stream_generator(actual_model_identifier=actual_model_identifier, langchain_messages=langchain_messages, request_temperature=request_data.temperature, request_max_tokens=request_data.max_tokens), media_type="text/event-stream",)
    else:
        history_messages = langchain_messages[:-1]; user_input_message_content = ""
        if isinstance(langchain_messages[-1], HumanMessage): user_input_message_content = langchain_messages[-1].content
        elif isinstance(langchain_messages[-1], SystemMessage) and len(langchain_messages) == 1: user_input_message_content = langchain_messages[-1].content
        else: raise HTTPException(status_code=400, detail="Last message must be 'user' or a single 'system' message.")
        input_data = {"history": history_messages, "input": user_input_message_content}
        llm_params: Dict[str, Any] = {"model": actual_model_identifier, "base_url": OLLAMA_BASE_URL}; default_temp = 0.7
        if hasattr(llm, 'temperature') and llm.temperature is not None: default_temp = llm.temperature
        llm_params["temperature"] = request_data.temperature if request_data.temperature is not None else default_temp
        if request_data.max_tokens is not None and request_data.max_tokens > 0: llm_params["num_predict"] = request_data.max_tokens
        try: current_llm = ChatOllama(**llm_params)
        except Exception as e: raise HTTPException(status_code=500, detail=f"Could not initialize LLM for model: {actual_model_identifier}. Error: {str(e)}")
        current_chain = prompt_template | current_llm | StrOutputParser()
        try: full_response_content = await current_chain.ainvoke(input_data)
        except Exception as e: raise HTTPException(status_code=500, detail=f"LLM processing error: {str(e)}")
        response_message = ChatMessageOutput(role="assistant", content=full_response_content); choice = ChatCompletionChoice(index=0, message=response_message, finish_reason="stop")
        usage_stats = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        return ChatCompletionResponse(model=actual_model_identifier, choices=[choice], usage=usage_stats)

@app.get("/") # (Keep as is)
async def root(): return {"message": f"Langchain Ollama OpenAI-Compatible API is running. Dummy model ID: '{DUMMY_MODEL_ID}'. Use /v1/models and /v1/chat/completions."}

# To run this application:
# 1. Save as `main.py`.
# 2. Ensure Ollama server is running and you have pulled models (e.g., `ollama pull llama3`).
# 3. Optionally, create a `.env` file for `OLLAMA_DEFAULT_MODEL`, `OLLAMA_BASE_URL`, `OPENWEBUI_ORIGIN`.
# 4. Install/Update dependencies: `pip install fastapi "uvicorn[standard]" python-dotenv langchain langchain-ollama ollama pydantic`
# 5. Run with Uvicorn: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
