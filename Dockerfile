# Imagen base ligera con Python 3.12
FROM python:3.12-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar primero requirements para aprovechar cache de Docker
COPY requirements.txt .

# Instalar dependencias (setuptools incluido para evitar el error de pkg_resources)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY src/ ./src/

# Exponer el puerto donde corre la API
EXPOSE 8000

# Comando para iniciar la API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]