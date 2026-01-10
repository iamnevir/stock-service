from fastapi import FastAPI
from fastapi.responses import FileResponse


app = FastAPI()


FILE_PATH = "/home/ubuntu/nevir/data/busd_last.pkl"


@app.get("/download/busd")
def download_busd():
    return FileResponse(
    path=FILE_PATH,
    filename="busd_last.pkl",
    media_type="application/octet-stream",
    )