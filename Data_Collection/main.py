"""
Face Data Collection — Main Entry Point

Usage:
    python main.py                          # Use Hikvision camera from .env
    python main.py --source webcam          # Use laptop webcam for testing
    python main.py --source path/to/video   # Use a video file for testing
    python main.py --no-dashboard           # Run without dashboard
    python main.py --config custom.yaml     # Use a custom config file
"""

import sys
import signal
import argparse
import logging
from pathlib import Path

# Add parent directory to path (for imports when running from Data_Collection/)
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.helpers import load_config, setup_logging
from src.collector import FaceCollector
from src.dashboard import create_dashboard


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Data Collection from CCTV / webcam / video file"
    )
    parser.add_argument(
        "--source", type=str, default=None,
        help="Video source: 'webcam', path to video file, or RTSP URL. "
             "If not specified, uses the Hikvision camera from .env"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    parser.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable the live monitoring dashboard"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Setup logging
    logger = setup_logging(level=args.log_level, log_dir="logs")

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Override source if specified
    if args.source is not None:
        if args.source.lower() == "webcam":
            config["camera"]["rtsp_url"] = 0  # OpenCV webcam index
            logger.info("Using webcam as video source")
        elif Path(args.source).exists():
            config["camera"]["rtsp_url"] = str(Path(args.source).resolve())
            logger.info(f"Using video file: {args.source}")
        else:
            # Assume it's an RTSP URL
            config["camera"]["rtsp_url"] = args.source
            logger.info(f"Using RTSP URL: {args.source}")

    # Disable dashboard if requested
    if args.no_dashboard:
        config["dashboard"]["enabled"] = False

    # Create collector
    collector = FaceCollector(config)

    # Create and attach dashboard
    dashboard = None
    if config.get("dashboard", {}).get("enabled", True):
        dashboard = create_dashboard(config)
        collector.set_dashboard(dashboard)

    # Graceful shutdown on Ctrl+C
    def signal_handler(sig, frame):
        logger.info("\nCtrl+C received — shutting down...")
        collector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Start everything
    try:
        collector.initialize()

        if dashboard:
            dashboard.start()
            port = config.get("dashboard", {}).get("port", 5000)
            logger.info(f"Dashboard running at http://localhost:{port}")

        logger.info("")
        logger.info("Press Ctrl+C to stop")
        logger.info("")

        collector.run()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
