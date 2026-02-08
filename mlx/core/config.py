# MLX Stack Manager - Core Configuration

from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """MLX stack configuration."""

    model_config = ConfigDict(env_prefix="MLX_")

    # Models
    chat_model: str = "mlx-community/Qwen3-VL-8B-Instruct-8bit"
    ocr_model: str = "mlx-community/GLM-OCR-8bit"
    asr_model: str = "mlx-community/Qwen3-ASR-1.7B-8bit"
    embed_model: str = "jedisct1/Qwen3-VL-Embedding-8B-mlx"

    # Ports
    chat_port: int = 8101
    ocr_port: int = 8102
    asr_port: int = 8103
    embed_port: int = 8100

    # Environment
    python_path: Path = Path("/opt/miniconda3/envs/mlx/bin/python")
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    log_dir: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "logs"
    )


settings = Settings()
settings.log_dir.mkdir(exist_ok=True)
