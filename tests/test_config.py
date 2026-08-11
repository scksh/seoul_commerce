"""프로젝트 설정 기능을 검증하는 테스트 모듈.

Author:
    SeungHwan Kim

Created:
    2026-08-11

Description:
    환경변수 및 .env 파일에서 API 인증키를 불러오는 동작을 검증한다.
"""


import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from seoul_commerce.config import SEOUL_API_KEY_NAME, load_api_key


class ApiKeyConfigTests(unittest.TestCase):
    def test_load_api_key_from_env_file(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            env_file = os.path.join(directory, ".env")
            with open(env_file, "w", encoding="utf-8") as file:
                file.write(f"{SEOUL_API_KEY_NAME}=test-key\n")

            self.assertEqual(load_api_key(env_file), "test-key")

    def test_process_environment_takes_precedence(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = os.path.join(directory, ".env")
            with open(env_file, "w", encoding="utf-8") as file:
                file.write(f"{SEOUL_API_KEY_NAME}=file-key\n")
            with patch.dict(os.environ, {SEOUL_API_KEY_NAME: "process-key"}, clear=True):
                self.assertEqual(load_api_key(env_file), "process-key")

    def test_missing_api_key_raises_clear_error(self) -> None:
        with TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, SEOUL_API_KEY_NAME):
                load_api_key(os.path.join(directory, "missing.env"))


if __name__ == "__main__":
    unittest.main()
