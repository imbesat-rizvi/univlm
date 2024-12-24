from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from typing import Dict,Any
from fastapi.middleware.cors import CORSMiddleware
from util.switch_conda_env import switch_conda_env


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8000"
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/generate")

async def generate_output_endpoint(model_name: str, image_path: str) -> Dict[str, str]:
    return await generate_output(model_name, image_path)

async def generate_output(model_name: str, image_path: str) -> Dict[str, str]:
    # Call other functions using model_name and image_path
    result = switch_conda_env(model_name)
    return {
        "model_name": model_name,
        "image_path": image_path,
        "result": result
    }

async def generate_output(model_name: str, image_path: str) -> Dict[str, str]:
    # Switch conda environment using model_name
    switch_conda_env(model_name)

#D:\!!realtest shit\images\DSC_0898.NEF