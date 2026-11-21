
from pathlib import Path

from huggingface_hub import hf_hub_download


MODEL_REPO = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

MODEL_FILENAME = "onnx/model_quint8_avx2.onnx"

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "embeddings_multilingue"

MODEL_PATH = MODEL_DIR / MODEL_FILENAME


def descargar_modelo():
    """Download the ONNX model when it is missing."""

    if MODEL_PATH.is_file():
        print("ONNX model already available.")
        return MODEL_PATH

    print("Downloading the ONNX embedding model...")

    ruta = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
        local_dir=str(MODEL_DIR),
    )

    if not Path(ruta).is_file():
        raise FileNotFoundError(f"ONNX model not found: {ruta}")

    print("Model downloaded successfully.")

    return Path(ruta)


if __name__ == "__main__":
    descargar_modelo()
