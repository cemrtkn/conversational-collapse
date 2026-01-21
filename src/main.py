import argparse
import logging
import os
from datetime import datetime
from typing import List

from babel_ai.experiment import Experiment, ExperimentConfig
from utils import load_yaml_config

logger = logging.getLogger(__name__)

# Remove the module-level logging.basicConfig() call
# This was preventing proper override in setup_logging()


def setup_logging(log_file: str = None, debug: bool = False):
    """Set up logging with proper configuration that can be overridden."""
    # Clear any existing handlers to allow reconfiguration
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Set up handlers
    handlers = []

    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    handlers.append(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,  # Force reconfiguration even if already configured
    )

    logger.info(f"Logging configured - Level: {'DEBUG' if debug else 'INFO'}")
    if log_file:
        logger.info(f"Log file: {log_file}")


def run_experiment(config: ExperimentConfig):
    """Run a single experiment and log its results."""
    experiment = Experiment(config)
    logger.info("Starting experiment thread.")
    experiment.run()


def run_experiment_batch(
    configs: List[ExperimentConfig],
    debug: bool = False,
):
    """Run multiple experiments sequentially."""

    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Set up logging with a more meaningful file name
    setup_logging(
        log_file=os.path.join(
            "logs",
            "experiment_{}.log".format(
                datetime.now().strftime("%Y%m%d_%H%M%S")
            ),
        ),
        debug=debug,
    )

    # Run experiments sequentially
    logger.info(f"Running {len(configs)} experiments sequentially.")
    for i, config in enumerate(configs):
        logger.info(
            f"Running experiment {i} with config: {config.model_dump()}"
        )
        run_experiment(config)
        logger.info(f"Experiment {i} completed")

    logging.info("All experiments completed.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run babel_ai experiments")
    parser.add_argument(
        "config_files", nargs="+", help="Path to experiment config files"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    # Set up basic console logging for startup
    setup_logging(debug=args.debug)

    # Load configs from provided file paths
    example_configs = []
    for config_path in args.config_files:
        config = load_yaml_config(ExperimentConfig, config_path)
        example_configs.append(config)

    run_experiment_batch(example_configs, debug=args.debug)
