import logging
import datetime


def create_logger(log_folder_path, log_stream_level="DEBUG", log_write_level="DEBUG"):
    """
    Création d'un logger pour la gestion des erreurs, bugs
    et du comportement du backend
    - Affiche les résultats dans la console
    - Création d'un fichier de log

    Args:
        log_folder_path (str) : Chemin du dossier où
                                stocker les fichiers de logs
        log_stream_level, log_write_level (str): niveaux standards des logs en streaming ou en écriture
    Returns:
        logging.logger : Logger pret à l'emploi pour l'application Flask
    """

    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level=log_stream_level)

    # Create handlers
    today = datetime.date.today()
    log_file_path = log_folder_path + "app_{}.log".format(today)
    f_handler = logging.FileHandler(log_file_path)
    c_handler = logging.StreamHandler()
    f_handler.setLevel(level=log_write_level)
    c_handler.setLevel(level=log_stream_level)

    # Create formatters and add it to handlers
    format_str = (
        "[%(asctime)s][%(levelname)1.1s][BACKEND_VISION360] "
        + "%(filename)s@%(lineno)d : %(message)s"
    )
    log_format = logging.Formatter(format_str)
    c_handler.setFormatter(log_format)
    f_handler.setFormatter(log_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger
