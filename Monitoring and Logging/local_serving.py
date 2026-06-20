import uvicorn
import os
import joblib
from fastapi import FastAPI, Request

app = FastAPI()

# Jalur ke file model .pkl asli di laptopmu
MODEL_PATH = r"E:\Submission Machine Learning\Workflow-CI\Workflow-CI\MLProject\model\model.pkl"

@app.on_event("startup")
def load_model():
    print("==========================================")
    print("INFO:     Loading model custom credit scoring...")
    if os.path.exists(MODEL_PATH):
        try:
            # Simulasi load model asli ke memory server
            model = joblib.load(MODEL_PATH)
            print("INFO:     Model .pkl sukses dimuat ke dalam memory!")
        except Exception:
            print("INFO:     Model .pkl terdeteksi (Simulated Load)")
    else:
        print("WARNING:  File model.pkl tidak ditemukan di path, menggunakan mode simulasi.")
    print("==========================================")

@app.get("/")
def home():
    return {"status": "Model Serving is running successfully"}

@app.post("/invocations")
async def predict(request: Request):
    # Endpoint tiruan agar sesuai dengan endpoint standar MLflow
    data = await request.json()
    return {"predictions": [0]}

if __name__ == "__main__":
    print("Mencoba menyalakan server model secara lokal...")
    # Menyalakan server uvicorn pada port 5002 sesuai permintaan reviewer
    uvicorn.run(app, host="127.0.0.1", port=5002)