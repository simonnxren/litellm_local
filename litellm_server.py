import os
import subprocess
import time
from pathlib import Path

# Colors
G = "\033[32m"
Y = "\033[33m"
R = "\033[31m"
B = "\033[34m"
N = "\033[0m"

PORT = os.getenv("ROUTER_PORT", "8200")

def compose(*args):
    """Run docker compose command for litellm."""
    try:
        subprocess.run(["docker", "compose", "-f", "docker-compose.litellm.yml", *args], check=True)
    except subprocess.CalledProcessError as e:
        print(f"{R}Error running docker compose: {e}{N}")

def is_enabled(service):
    """Check if a service is enabled in .env."""
    return os.getenv(f"VLLM_{service.upper()}_ENABLE", "false").lower() == "true"

def generate_config():
    """Generate litellm_config.yaml from environment variables (enabled services only)."""
    models = []
    
    # Embedding
    if is_enabled("embed"):
        embed_model = os.getenv("MODEL_EMBED_NAME", "Qwen/Qwen3-VL-Embedding-2B")
        models.append(f"""  # EMBEDDING SERVICE
  - model_name: embedding
    litellm_params:
      model: hosted_vllm/{embed_model}
      api_base: http://vllm-embedding:8100/v1
      api_key: "not-needed"
    model_info:
      mode: embedding

  - model_name: {embed_model}
    litellm_params:
      model: hosted_vllm/{embed_model}
      api_base: http://vllm-embedding:8100/v1
      api_key: "not-needed"
    model_info:
      mode: embedding
""")

    # Completions
    if is_enabled("completions"):
        comp_model = os.getenv("MODEL_COMPLETIONS_NAME", "Qwen/Qwen3-VL-8B-Instruct")
        models.append(f"""  # COMPLETIONS SERVICE
  - model_name: completions
    litellm_params:
      model: hosted_vllm/{comp_model}
      api_base: http://vllm-completions:8101/v1
      api_key: "not-needed"
      supports_response_schema: true
    model_info:
      mode: chat
      supports_function_calling: true

  - model_name: {comp_model}
    litellm_params:
      model: hosted_vllm/{comp_model}
      api_base: http://vllm-completions:8101/v1
      api_key: "not-needed"
      supports_response_schema: true
      reasoning_parser: deepseek_r1
    model_info:
      mode: chat
""")

    # OCR
    if is_enabled("ocr"):
        ocr_model = os.getenv("MODEL_OCR_NAME", "tencent/HunyuanOCR")
        models.append(f"""  # OCR SERVICE
  - model_name: ocr
    litellm_params:
      model: hosted_vllm/{ocr_model}
      api_base: http://vllm-ocr:8102/v1
      api_key: "not-needed"
    model_info:
      mode: chat
      supports_vision: true

  - model_name: {ocr_model}
    litellm_params:
      model: hosted_vllm/{ocr_model}
      api_base: http://vllm-ocr:8102/v1
      api_key: "not-needed"
    model_info:
      mode: chat
      supports_vision: true
""")

    # ASR
    if is_enabled("asr"):
        asr_model = os.getenv("MODEL_ASR_NAME", "openai/whisper-large-v3-turbo")
        models.append(f"""  # ASR SERVICE
  - model_name: asr
    litellm_params:
      model: hosted_vllm/{asr_model}
      api_base: http://vllm-asr:8103/v1
      api_key: "not-needed"
    model_info:
      mode: audio_transcription

  - model_name: {asr_model}
    litellm_params:
      model: hosted_vllm/{asr_model}
      api_base: http://vllm-asr:8103/v1
      api_key: "not-needed"
    model_info:
      mode: audio_transcription
""")

    if not models:
        print(f"  {R}✗ No services enabled{N}")
        return False
    
    config = f"""# LiteLLM Proxy Configuration (auto-generated from .env)

model_list:
{"".join(models)}
router_settings:
  routing_strategy: least-busy
  num_retries: 3
  timeout: 300
  allowed_fails: 1
  cooldown_time: 10

litellm_settings:
  drop_params: true
  request_timeout: 300
  telemetry: false
  max_parallel_requests: 100
  cache: true
  cache_params:
    type: "local"
    ttl: 3600

general_settings:
  store_model_in_db: false
  max_request_size_mb: 100
"""
    Path("litellm_config.yaml").write_text(config)
    print(f"  {G}✓{N} Generated litellm_config.yaml")
    return True

def start():
    """Start LiteLLM gateway."""
    print(f"{B}Starting LiteLLM gateway...{N}")
    if not generate_config():
        return
    compose("up", "-d")
    print(f"{G}✓ Gateway started{N}")

def stop():
    """Stop LiteLLM gateway."""
    print(f"{B}Stopping LiteLLM gateway...{N}")
    compose("down")

def status():
    """Check gateway status."""
    try:
        import requests
        resp = requests.get(f"http://localhost:{PORT}/health")
        if resp.status_code == 200:
            print(f"{G}✓ Gateway UP{N} (:{PORT})")
        else:
            print(f"{Y}⚠ Gateway ISSUES{N} (Status: {resp.status_code})")
    except:
        print(f"{R}✗ Gateway DOWN{N}")

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start": start()
    elif cmd == "stop": stop()
    elif cmd == "status": status()
    else: print("Usage: python litellm_server.py {start|stop|status}")
