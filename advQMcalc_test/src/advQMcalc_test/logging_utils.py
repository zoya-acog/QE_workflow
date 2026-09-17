import logging
from pathlib import Path

def setup_logger(
        name: str = "advQMcalc_test",
        log_file: Path | None = None,
        ) -> logging.Logger:
    """
    setup a logger that logs to the console and optionally to a file.
    if log_file is provided, logs are appended to that file.
    """

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            )

    # console handler (add only once)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)

        if not any(
                isinstance(h, logging.FileHandler) and
                h.baseFilename == str(log_file) for h in logger.handlers
                ):
            file_handler = logging.FileHandler(log_file, mode="a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
