# MLX Service Manager - API Routes

from datetime import datetime
from fastapi import APIRouter, HTTPException

from api.models import ServiceListResponse, ServiceStatus, ActionResponse
from core.service_manager import ServiceManager

router = APIRouter()
manager = ServiceManager()


@router.get("/health")
async def health():
    """Manager health check and aggregate service health."""
    services = manager.list_services()
    healthy_count = sum(1 for s in services if s.healthy)
    return {
        "status": "healthy",
        "services_healthy": f"{healthy_count}/{len(services)}",
        "services": [{"name": s.name, "healthy": s.healthy} for s in services],
    }


@router.get("/services", response_model=ServiceListResponse)
async def list_services():
    """List all services with their status."""
    return ServiceListResponse(
        services=manager.list_services(), timestamp=datetime.now()
    )


@router.get("/services/{service}/status", response_model=ServiceStatus)
async def get_service_status(service: str):
    """Get status of a specific service."""
    try:
        return manager.get_status(service)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/services/{service}/start", response_model=ActionResponse)
async def start_service(service: str):
    """Start a specific service."""
    try:
        status = manager.start_service(service)
        return ActionResponse(
            success=True, message=f"{service} started", service=status
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/services/{service}/stop", response_model=ActionResponse)
async def stop_service(service: str):
    """Stop a specific service."""
    try:
        status = manager.stop_service(service)
        return ActionResponse(
            success=True, message=f"{service} stopped", service=status
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/services/{service}/restart", response_model=ActionResponse)
async def restart_service(service: str):
    """Restart a specific service."""
    try:
        status = manager.restart_service(service)
        return ActionResponse(
            success=True, message=f"{service} restarted", service=status
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/services/{service}/logs")
async def get_service_logs(service: str, lines: int = 100):
    """Get logs from a specific service."""
    try:
        log_lines = manager.get_logs(service, lines)
        return {"service": service, "lines": log_lines}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/services/start-all", response_model=ServiceListResponse)
async def start_all_services():
    """Start all services."""
    return ServiceListResponse(services=manager.start_all(), timestamp=datetime.now())


@router.post("/services/stop-all", response_model=ServiceListResponse)
async def stop_all_services():
    """Stop all services."""
    return ServiceListResponse(services=manager.stop_all(), timestamp=datetime.now())
