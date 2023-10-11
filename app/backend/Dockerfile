# Dockerfile for Backend (FastAPI)

# Use a base image with Python and other dependencies
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app/backend

# Copy the FastAPI app files into the container
COPY .

# Install any necessary dependencies for the backend
RUN pip install --no-cache-dir -r backend_requirement.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
