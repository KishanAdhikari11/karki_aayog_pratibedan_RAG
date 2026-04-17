from slowapi import Limiter
from slowapi.util import get_remote_address
from config import config

RATE_LIMIT = "10/minute"
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=config.REDIS_URL,
    default_limits=[RATE_LIMIT],
)
