import os
from yapeco import BaseEnvironment as BaseEnv

if os.path.exists(".env"):
    from dotenv import load_dotenv

    load_dotenv()


class Env(BaseEnv):
    # remote inference API access
    orpheus_api_url: str
    orpheus_api_timeout: int = 16
    # inference settings
    orpheus_max_tokens: int = 8192
    orpheus_temperature: float = 0.7
    orpheus_top_p: float = 0.95
    orpheus_sample_rate: int = 24000
