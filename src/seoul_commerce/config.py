"""서울시 상권 분석 프로젝트의 설정 모듈.

Author:
    SeungHwan Kim

Created:
    2026-08-11

Description:
    프로젝트 환경변수를 불러오고 서울 OpenAPI 인증키를 관리한다.
"""

import os

from dotenv import dotenv_values, load_dotenv


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DEFAULT_ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
SEOUL_API_KEY_NAME = "SEOUL_OPEN_DATA_API_KEY"

# 프로젝트 루트의 .env 값을 환경변수로 불러온다.
try:
    load_dotenv(DEFAULT_ENV_FILE)
except (OSError, UnicodeError) as error:
    raise ValueError(f".env 파일을 읽을 수 없습니다: {DEFAULT_ENV_FILE}") from error


def load_api_key(env_file: str = DEFAULT_ENV_FILE) -> str:
    """
    서울 OpenAPI 인증키를 불러온다.

    Args:
        env_file: 인증키가 작성된 환경변수 파일 경로

    Returns:
        서울 OpenAPI 인증키

    Raises:
        ValueError: 환경변수와 .env 파일에 인증키가 없는 경우
    """
    api_key = os.getenv(SEOUL_API_KEY_NAME, "").strip()

    if not api_key and os.path.isfile(env_file):
        try:
            api_key = str(
                dotenv_values(env_file).get(SEOUL_API_KEY_NAME, "")
            ).strip()
        except (OSError, UnicodeError) as error:
            raise ValueError(f".env 파일을 읽을 수 없습니다: {env_file}") from error

    if not api_key:
        raise ValueError(
            f"{SEOUL_API_KEY_NAME}가 설정되지 않았습니다. "
            f"{env_file}에 인증키를 입력해주세요."
        )

    return api_key
