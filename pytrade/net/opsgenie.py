from typing import Optional, Any, Collection, Dict

from pytrade.net.http import _send_request, HttpMethod, HttpRequest
from pytrade.utils.profile import load_profile
from pytrade.utils.retry import retry
from time import sleep

BASE_URL = "https://api.opsgenie.com"

APP_URL = "https://app.opsgenie.com"


def create_alert(message: str, description: str, priority: Optional[str] = None,
                 tags: Optional[Collection[str]] = None,
                 details: Optional[Dict[str, Any]] = None,
                 actions: Optional[Collection[str]] = None,
                 profile: Optional[str] = None) -> Dict:
    profile = load_profile(profile)
    headers = {
        "Authorization": f"GenieKey {profile.opsgenie_api_key}",
    }
    json = {"message": message, "description": description}
    if priority is not None:
        json["priority"] = priority
    if tags is not None:
        json["tags"] = tags
    if details is not None:
        json["details"] = details
    if actions is not None:
        json["actions"] = actions
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint="/v2/alerts",
        method=HttpMethod.POST,
        headers=headers,
        json=json
    )
    res = retry(_send_request, args=(req,))
    return res.json()


def get_request_status(request_id, profile: Optional[str] = None):
    profile = load_profile(profile)
    headers = {
        "Authorization": f"GenieKey {profile.opsgenie_api_key}",
    }
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=f"/v2/alerts/requests/{request_id}",
        headers=headers
    )
    return retry(_send_request, args=(req,)).json()


def get_alert_id(request_id: str, profile: Optional[str] = None) -> str:
    status = get_request_status(request_id, profile)
    return status["data"]["alertId"]


def get_alert_url(alert_id: str) -> str:
    return f"{APP_URL}/alert/detail/{alert_id}/details"


def get_alert(alert_id: str, profile: Optional[str] = None) -> Dict:
    profile = load_profile(profile)
    headers = {
        "Authorization": f"GenieKey {profile.opsgenie_api_key}",
    }
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=f"/v2/alerts/{alert_id}",
        headers=headers
    )
    return retry(_send_request, args=(req,)).json()


def wait_until_closed(alert_id: str, interval: int = 10,
                      timeout: Optional[int] = None,
                      profile: Optional[str] = None) -> None:
    total_time = 0
    while True:
        alert = get_alert(alert_id, profile)
        if alert["data"]["status"] == "closed":
            return
        sleep(interval)
        total_time += interval
        if timeout is not None and total_time > timeout:
            break
    raise ValueError("Error waiting until closed; timeout")
