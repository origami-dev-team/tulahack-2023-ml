from fastapi import FastAPI, APIRouter, UploadFile, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
from utils import generate_image
from torch import load

router = APIRouter()
model = load("./notebooks/model.pth")

class GenerateDTO(BaseModel):
    prompt: str

@router.post("/background")
def generate_background(generate_dto: GenerateDTO):
    images = generate_image(query=generate_dto.prompt, model=model, generate_emotions=False)
    background = images[0]
    print(background)
    Response(content=background, media_type="application/png")
    

@router.post("/character")
def generate_character(generate_dto: GenerateDTO) -> int:
    images = generate_image(query=generate_dto.prompt, model=model, generate_emotions=True)
    varinats = ["default", "happy", "sad", "angry"]
    files = []
    for i, image in enumerate(images):
        files.append(UploadFile(file=image, filename=f"{varinats[i]}.png"))
    return len(images)


app = FastAPI()
app.include_router(router, prefix="/generate")
