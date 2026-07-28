# Use a lightweight Python image
FROM python:3.10-slim

# HF Spaces mandates running as a non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR $HOME/app

# Copy requirements first to leverage Docker layer caching
COPY --chown=user:user requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY --chown=user:user . .

# Expose the specific port Hugging Face listens to
EXPOSE 7860

# Launch the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]