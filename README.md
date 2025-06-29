## ¿Cómo funciona el repositorio?

1. **Integración continua**
   - El workflow `integration.yml`:
     - Se ejecuta en cada Pull Request o manualmente.
     - Instala dependencias y ejecuta los tests unitarios de `unit_tests/` con cobertura.
     - Publica un comentario automático en el Pull Request con los resultados y cobertura.
     - Falla si alguna prueba no pasa, asegurando la calidad antes de fusionar cambios.

2. **Entrenamiento y registro del modelo**
   - El workflow `build.yml` se encarga de:
     - Descargar los datos.
     - Entrenar el modelo (`src/main.py`).
     - Guardar el modelo entrenado en `models/model.pkl`.
     - Ejecutar pruebas unitarias sobre el modelo (`model_tests/test_model.py`).
     - Registrar el modelo en MLflow (`scripts/register_model.py`).

3. **Despliegue del modelo**
   - El workflow `deploy.yml`:
     - Construye una imagen Docker con la API del modelo.
     - Sube la imagen a Azure Container Registry.
     - Despliega la imagen en Azure Container Instances como un servicio REST.

## Requisitos previos

- Configurar los siguientes secretos y variables en GitHub Actions:
  - Secrets: `AZURE_CREDENTIALS`, `ACR_NAME`, `ACR_USERNAME`, `ACR_PASSWORD`, `AZURE_RESOURCE_GROUP`, `AZURE_STORAGE_CONNECTION_STRING`
  - Variables: `EXPERIMENT_NAME`, `MLFLOW_URL`, `MODEL_NAME`

## Ejecución local

1. Instala las dependencias:
    ```bash
    pip install -r requirements.txt
    ```
   
3. Entrena el modelo:
   ```bash
   python src/main.py
   ```
4. Ejecuta las pruebas:
    ```bash
    pytest model_tests/test_model.py
    ```

## Despliegue automático

- El despliegue y entrenamiento se realiza automáticamente al hacer push a la rama `main` o ejecutando manualmente los workflows desde GitHub Actions.

## Consulta a la API

- Modifica la URL en `scripts/query_model.py` si es necesario y ejecuta:
    ```bash
    python scripts/query_model.py
    ```
   