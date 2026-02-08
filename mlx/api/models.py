# MLX Service Manager - Pydantic Models

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class ServiceState(str, Enum):
    RUNNING = "running"
    STOPPED = "stopped"


class ServiceStatus(BaseModel):
    name: str
    state: ServiceState
    pid: Optional[int] = None
    uptime: Optional[float] = None
    port: int
    healthy: bool = False
    log_file: str


class ServiceListResponse(BaseModel):
    services: List[ServiceStatus]
    timestamp: datetime


class ActionResponse(BaseModel):
    success: bool
    message: str
    service: Optional[ServiceStatus] = None
