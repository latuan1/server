import uvicorn
from src.api.api import app

if __name__ == "__main__":
    # Chạy server trên host 0.0.0.0 và port 8000
    uvicorn.run(app, host="0.0.0.0", port=5000)