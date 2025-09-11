from pytrade.net.http import HttpRequest, _send_request

BASE_URL = "https://blockstream.info/api"


def get_height():
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint="/blocks/tip/height"
    )
    res = _send_request(req)
    return int(res.text)
