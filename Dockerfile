# Server-only (OpenAI-compatible endpoints). Mic/speaker pipeline runs natively on the Mac.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y libsndfile1 espeak-ng && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pip install -e ".[kokoro]"
EXPOSE 8000
CMD ["python", "-m", "josie", "serve", "--host", "0.0.0.0"]
