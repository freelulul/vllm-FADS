# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, Response

import vllm.envs as envs
from vllm.engine.protocol import EngineClient
from vllm.logger import init_logger

logger = init_logger(__name__)


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


router = APIRouter()


@router.post("/sleep")
async def sleep(raw_request: Request):
    # get POST params
    level = raw_request.query_params.get("level", "1")
    mode = raw_request.query_params.get("mode", "abort")
    await engine_client(raw_request).sleep(int(level), mode)
    # FIXME: in v0 with frontend multiprocessing, the sleep command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


@router.post("/wake_up")
async def wake_up(raw_request: Request):
    tags = raw_request.query_params.getlist("tags")
    if tags == []:
        # set to None to wake up all tags if no tags are provided
        tags = None
    logger.info("wake up the engine with tags: %s", tags)
    await engine_client(raw_request).wake_up(tags)
    # FIXME: in v0 with frontend multiprocessing, the wake-up command
    # is sent but does not finish yet when we return a response.
    return Response(status_code=200)


@router.post("/reload_tokenizer")
async def reload_tokenizer(raw_request: Request):
    """Reload the tokenizer after a cross-GPU model migration.

    After reload_for_migration swaps the model weights in the EngineCore
    worker, the APIServer still holds the OLD tokenizer. This endpoint
    swaps the tokenizer in-place on the existing renderer, input_processor,
    and output_processor — without replacing the objects (the background
    output_handler holds closures over them).
    """
    model_path = raw_request.query_params.get("model_path")
    if not model_path:
        return JSONResponse(content={"error": "model_path required"}, status_code=400)

    client = engine_client(raw_request)

    # Build a new renderer just to get the new tokenizer
    client.vllm_config.model_config.tokenizer = model_path
    from vllm.renderers.registry import renderer_from_config
    new_renderer = renderer_from_config(client.vllm_config)

    # Swap tokenizer on ALL objects that hold a renderer reference.
    # vLLM copies the renderer to multiple places at init time.
    client.renderer = new_renderer
    client.input_processor.renderer = new_renderer
    client.input_processor.input_preprocessor.renderer = new_renderer
    client.output_processor.tokenizer = new_renderer.tokenizer

    # OpenAI serving layer also copies the renderer
    app = raw_request.app
    if hasattr(app.state, 'openai_serving_completion') and app.state.openai_serving_completion:
        app.state.openai_serving_completion.renderer = new_renderer
    if hasattr(app.state, 'openai_serving_chat') and app.state.openai_serving_chat:
        app.state.openai_serving_chat.renderer = new_renderer
    if hasattr(app.state, 'openai_serving_render') and app.state.openai_serving_render:
        app.state.openai_serving_render.renderer = new_renderer

    logger.info("Tokenizer reloaded for model: %s", model_path)
    return JSONResponse(content={"status": "ok", "model_path": model_path})


@router.get("/is_sleeping")
async def is_sleeping(raw_request: Request):
    is_sleeping = await engine_client(raw_request).is_sleeping()
    return JSONResponse(content={"is_sleeping": is_sleeping})


def attach_router(app: FastAPI):
    if not envs.VLLM_SERVER_DEV_MODE:
        return

    app.include_router(router)
