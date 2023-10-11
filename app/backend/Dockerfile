# Dockerfile for Backend (FastAPI)

# Use a base image with Python and other dependencies
FROM python:3.8-slim-buster

# Set the working directory
WORKDIR /app/backend

# Copy the FastAPI app files into the container
COPY . /app/backend/

# Install any necessary dependencies for the backend
RUN pip install --no-cache-dir -r requirement.txt

# Expose the FastAPI port
# Expose the Streamlit port
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "main.py"]
