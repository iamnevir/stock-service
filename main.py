import os
import shutil
from fastapi import FastAPI, BackgroundTasks, HTTPException
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


@app.get("/download/mongo_bak/{date}")
def download_mongo_bak(date: str, background_tasks: BackgroundTasks):
    folder_path = f"/home/ubuntu/nevir/mongo_bak/{date}"
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        raise HTTPException(status_code=404, detail="Backup folder not found")
    
    zip_filename = f"/tmp/mongo_bak_{date}"
    zip_filepath = f"{zip_filename}.zip"
    
    shutil.make_archive(zip_filename, 'zip', folder_path)
    
    background_tasks.add_task(os.remove, zip_filepath)
    
    return FileResponse(
        path=zip_filepath,
        filename=f"mongo_bak_{date}.zip",
        media_type="application/zip",
    )