from pydantic import BaseModel
from fastapi import FastAPI, BackgroundTasks
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from typing import List, Dict

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class TrainConfig(BaseModel):
    model_name: str = "Score-Transformer"
    dataset: str = "/demo"
    steps: int = 1000
    base_model: False or str = False

train_config = TrainConfig()

@app.get("/")
def home():
    return FileResponse('index.html')

@app.get("/api/models")
async def list_base_models():
    return [    
        {"name": "Latent drummer",                  "id": "latent_drummer"}, 
        {"name": "Rhythm Transformer",              "id": "rhythm_transformer"}, 
        {"name": "Score-Transformer",               "id": "score_transformer"}, 
        {"name": "What How Play Auxillary VAE",     "id": "whp_avae"}, 
        {"name": "Convincing Harmony ",             "id": "convincing_harmony_lstm"}, 
        {"name": "Bach Duet ",                      "id": "bach_duet_stack_mem_lstm"}, 
        {"name": "PIIRNN",                          "id": "pi_rnn"}, 
        {"name": "Drum RBM",                        "id": "drum_rbm"}, 
    ]

@app.get("/api/config/{model}")
def get_config(model):
    global train_config
    train_config.model_name = model
    return train_config

@app.post("/api/config")
def post_config(new_config:TrainConfig):
    global train_config
    train_config = new_config
    return train_config

@app.get("/api/train")
async def get_train():
    return

@app.post("/api/train")
async def post_train():
    jobid= "ok"
    return jobid

@app.get("/api/jobs/{id}")
async def query_job(id):
    return id
