from contextlib import contextmanager
import os


@contextmanager
def no_proxy_env():
    """
    临时移除代理环境变量，避免 requests/akshare 因代理不可用而失败。
    """
    keys = [
        "HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
        "ALL_PROXY", "all_proxy", "NO_PROXY", "no_proxy",
    ]
    backup = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            if k in os.environ:
                os.environ.pop(k, None)
        os.environ["NO_PROXY"] = "*"
        os.environ["no_proxy"] = "*"
        yield
    finally:
        for k, v in backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
