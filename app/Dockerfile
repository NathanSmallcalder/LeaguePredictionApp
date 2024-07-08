# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt



CMD [ "python", "app.py"]