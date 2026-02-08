#!/usr/bin/env python3
"""
MLX Service Manager — Unified CLI and FastAPI orchestrator for MLX stack.

Usage:
    python manager.py start [service]    # Start services
    python manager.py stop [service]     # Stop services
    python manager.py status             # Show status
    python manager.py server             # Start API server
"""

import argparse
import json
import sys
from datetime import datetime

import uvicorn

from core.service_manager import ServiceManager
from api.models import ServiceState

# Service choices
SERVICE_CHOICES = ["embedding", "chat", "ocr", "asr"]

# CLI Manager instance
manager = ServiceManager()

# ---------------------------------------------------------------------------
# CLI Functions
# ---------------------------------------------------------------------------


def cli_action(args, action: str):
    """Generic CLI action handler."""
    service = getattr(args, "service", None)

    if service in (None, "all"):
        method = f"{action}_all"
        results = getattr(manager, method)()
        for svc in results:
            symbol = "✓" if svc.state == ServiceState.RUNNING else "✗"
            extra = f"(pid={svc.pid})" if svc.pid else ""
            print(f"{symbol} {svc.name} {action}d {extra}")
    else:
        result = getattr(manager, f"{action}_service")(service)
        symbol = "✓" if result.state == ServiceState.RUNNING else "✗"
        extra = f"(pid={result.pid})" if result.pid else ""
        print(f"{symbol} {result.name} {action}d {extra}")


def cli_start(args):
    cli_action(args, "start")


def cli_stop(args):
    cli_action(args, "stop")


def cli_restart(args):
    cli_action(args, "restart")


def cli_status(args):
    services = manager.list_services()
    print(
        json.dumps(
            {
                "services": [s.model_dump() for s in services],
                "timestamp": datetime.now().isoformat(),
            },
            indent=2,
            default=str,
        )
    )


def cli_health(args):
    services = manager.list_services()
    healthy_count = sum(1 for s in services if s.healthy)
    print(f"Services: {healthy_count}/{len(services)} healthy\n")
    for svc in services:
        symbol = "✓" if svc.healthy else "✗"
        print(f"  {symbol} {svc.name:12} {svc.state.value:10} port={svc.port}")


def cli_logs(args):
    lines = manager.get_logs(args.service, args.lines)
    for line in lines:
        print(line, end="")


def cli_server(args):
    import logging

    logger = logging.getLogger("mlx-manager")
    logger.info("Starting MLX Service Manager API on %s:%d", args.host, args.port)
    logger.info("API docs: http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        "api.app:app" if args.reload else "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="MLX Service Manager - Unified CLI and API server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Start command
    start_parser = subparsers.add_parser("start", help="Start services")
    start_parser.add_argument(
        "service", nargs="?", default="all", choices=["all"] + SERVICE_CHOICES
    )
    start_parser.set_defaults(func=cli_start)

    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop services")
    stop_parser.add_argument(
        "service", nargs="?", default="all", choices=["all"] + SERVICE_CHOICES
    )
    stop_parser.set_defaults(func=cli_stop)

    # Restart command
    restart_parser = subparsers.add_parser("restart", help="Restart service")
    restart_parser.add_argument("service", choices=SERVICE_CHOICES)
    restart_parser.set_defaults(func=cli_restart)

    # Status command
    status_parser = subparsers.add_parser("status", help="Show service status")
    status_parser.set_defaults(func=cli_status)

    # Health command
    health_parser = subparsers.add_parser("health", help="Health check")
    health_parser.set_defaults(func=cli_health)

    # Logs command
    logs_parser = subparsers.add_parser("logs", help="Show service logs")
    logs_parser.add_argument("service", choices=SERVICE_CHOICES)
    logs_parser.add_argument("lines", type=int, nargs="?", default=100)
    logs_parser.set_defaults(func=cli_logs)

    # Server command
    server_parser = subparsers.add_parser("server", help="Start FastAPI server")
    server_parser.add_argument("--host", default="0.0.0.0")
    server_parser.add_argument("--port", type=int, default=9000)
    server_parser.add_argument("--reload", action="store_true")
    server_parser.set_defaults(func=cli_server)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
