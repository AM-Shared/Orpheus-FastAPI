dev:
    uvicorn app:app --port 5005 --reload

build:
    docker build . -t ghcr.io/am-shared/orpheus-fastapi:latest
