FROM python:3.10-slim
LABEL authors="Admin"

ENTRYPOINT ["top", "-b"]

# Thiết lập thư mục làm việc trong container
WORKDIR /app

# Sao chép file requirements.txt vào container
COPY requirements.txt .

# Cài đặt thư viện từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép toàn bộ code vào container
COPY . .

# Mở cổng 80 (hoặc cổng bạn muốn)
EXPOSE 5000

# Chạy ứng dụng FastAPI bằng Uvicorn
CMD ["python", "main.py"]