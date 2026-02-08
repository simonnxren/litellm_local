# MLX Service Manager - Business Logic

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import psutil  # type: ignore

from core.config import settings
from api.models import ServiceState, ServiceStatus

logger = logging.getLogger("mlx-manager")


@dataclass
class ServiceConfig:
    """Service configuration."""

    name: str
    port: int
    cmd: List[str]
    health_path: str

    @property
    def log_file(self) -> Path:
        return settings.log_dir / f"{self.name}.log"


class ServiceManager:
    """Manages MLX service processes."""

    def __init__(self):
        self.processes: dict[str, psutil.Process] = {}
        self.start_times: dict[str, float] = {}
        self.services = self._init_services()

    def _init_services(self) -> dict[str, ServiceConfig]:
        """Initialize service configurations."""
        python = str(settings.python_path)
        project_root = settings.project_root

        return {
            "embedding": ServiceConfig(
                name="embedding",
                port=settings.embed_port,
                cmd=[
                    python,
                    str(project_root / "services" / "embedding_server.py"),
                    "--model",
                    settings.embed_model,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(settings.embed_port),
                ],
                health_path="/health",
            ),
            "chat": ServiceConfig(
                name="chat",
                port=settings.chat_port,
                cmd=[
                    python,
                    "-m",
                    "mlx_lm.server",
                    "--model",
                    settings.chat_model,
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(settings.chat_port),
                ],
                health_path="/v1/models",
            ),
            "ocr": ServiceConfig(
                name="ocr",
                port=settings.ocr_port,
                cmd=[
                    python,
                    "-m",
                    "mlx_vlm.server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(settings.ocr_port),
                ],
                health_path="/health",
            ),
            "asr": ServiceConfig(
                name="asr",
                port=settings.asr_port,
                cmd=[
                    python,
                    "-m",
                    "mlx_audio.server",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(settings.asr_port),
                    "--workers",
                    "1",
                ],
                health_path="/",
            ),
        }

    def _validate_service(self, name: str) -> ServiceConfig:
        """Validate service name and return config."""
        if name not in self.services:
            raise ValueError(f"Unknown service: {name}")
        return self.services[name]

    def is_running(self, name: str) -> bool:
        """Check if service process is running."""
        if name not in self.processes:
            return False
        try:
            return self.processes[name].is_running()
        except psutil.NoSuchProcess:
            self.processes.pop(name, None)
            return False

    def start_service(self, name: str) -> ServiceStatus:
        """Start a service."""
        config = self._validate_service(name)

        if self.is_running(name):
            logger.info("%s already running", name)
            return self.get_status(name)

        logger.info("Starting %s on port %d", name, config.port)

        with open(config.log_file, "a") as f:
            proc = subprocess.Popen(
                config.cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )

        self.processes[name] = psutil.Process(proc.pid)
        self.start_times[name] = time.time()
        logger.info("%s started (pid=%d)", name, proc.pid)

        return self.get_status(name)

    def stop_service(self, name: str) -> ServiceStatus:
        """Stop a service."""
        self._validate_service(name)

        if not self.is_running(name):
            logger.info("%s not running", name)
            return self.get_status(name)

        proc = self.processes[name]
        logger.info("Stopping %s (pid=%d)", name, proc.pid)

        proc.terminate()
        try:
            proc.wait(timeout=5)
        except psutil.TimeoutExpired:
            logger.warning("%s didn't stop gracefully, killing", name)
            proc.kill()
            proc.wait()

        self.processes.pop(name, None)
        self.start_times.pop(name, None)
        logger.info("%s stopped", name)

        return self.get_status(name)

    def restart_service(self, name: str) -> ServiceStatus:
        """Restart a service."""
        if self.is_running(name):
            self.stop_service(name)
            time.sleep(1)
        return self.start_service(name)

    def get_status(self, name: str) -> ServiceStatus:
        """Get service status."""
        config = self._validate_service(name)
        is_running = self.is_running(name)

        if is_running:
            proc = self.processes[name]
            return ServiceStatus(
                name=name,
                state=ServiceState.RUNNING,
                pid=proc.pid,
                uptime=time.time() - self.start_times.get(name, time.time()),
                port=config.port,
                healthy=self._check_health(config),
                log_file=str(config.log_file),
            )
        else:
            return ServiceStatus(
                name=name,
                state=ServiceState.STOPPED,
                port=config.port,
                log_file=str(config.log_file),
            )

    def _check_health(self, config: ServiceConfig) -> bool:
        """HTTP health check."""
        import requests  # type: ignore

        try:
            resp = requests.get(
                f"http://localhost:{config.port}{config.health_path}", timeout=1
            )
            return resp.status_code < 500
        except Exception:
            return False

    def get_logs(self, name: str, lines: int = 100) -> List[str]:
        """Get last N lines from service log."""
        config = self._validate_service(name)
        if not config.log_file.exists():
            return []
        with open(config.log_file, "r") as f:
            return f.readlines()[-lines:]

    def list_services(self) -> List[ServiceStatus]:
        """Get status of all services."""
        return [self.get_status(name) for name in self.services.keys()]

    def start_all(self) -> List[ServiceStatus]:
        """Start all services."""
        return self._apply_to_all("start_service", list(self.services.keys()))

    def stop_all(self) -> List[ServiceStatus]:
        """Stop all services."""
        return self._apply_to_all("stop_service", list(self.processes.keys()))

    def _apply_to_all(self, method_name: str, names: List[str]) -> List[ServiceStatus]:
        """Apply method to all service names."""
        results = []
        for name in names:
            try:
                results.append(getattr(self, method_name)(name))
            except Exception as e:
                logger.error("Failed to %s %s: %s", method_name, name, e)
        return results
