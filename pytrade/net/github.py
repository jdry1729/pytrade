import logging
from typing import Optional, Iterable, Dict

import numpy as np
import pandas as pd
from pytrade.net.http import HttpRequest, send_request, HttpMethod
from pytrade.utils.collections import get

logger = logging.getLogger(__name__)

BASE_URL = "https://api.github.com"

STARGAZERS_QUERY = """
query($owner: String!, $name:String!, $after: String) {
  repository(owner: $owner, name: $name) {
    stargazers(
      first: 100
      after: $after
      orderBy: {field: STARRED_AT, direction: ASC}
    ) {
      edges {
        node {
          login
        }
        cursor
        starredAt
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
}
"""

PULL_REQUESTS_QUERY = """
query ($owner: String!, $name: String!, $after: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      states: [MERGED]
      first: 100
      after: $after
      orderBy: {field: CREATED_AT, direction: ASC}
    ) {
      edges {
        node {
          createdAt
          mergedAt
          author {
            login
          }
          additions
          deletions
          number
        }
        cursor
      }
      pageInfo {
        endCursor
        hasNextPage
      }
    }
  }
  rateLimit {
    limit
    cost
    remaining
    resetAt
  }
}
"""


def _get_login(author):
    if author is None:
        return np.nan
    return author["login"]


def _get_objs(query: str, path: Iterable[str],
              variables: Optional[Dict[str, str]] = None,
              cursor: Optional[str] = None,
              auth_token: Optional[str] = None) -> Iterable[Dict]:
    """
    Gets all objects for a particular query.

    Parameters
    ----------
    query
        Query. Must return rate limit field.
    path
        Path to list of objects.
    variables
        Variables to pass to the query.
    cursor
        Optional cursor to start from.
    auth_token
        Auth token.

    Returns
    -------
    List of objects.
    """
    # TODO: use frozendict instead?
    variables = variables.copy()

    headers = {}
    if auth_token is not None:
        headers = {"Authorization": f"Bearer {auth_token}"}

    i = 0
    objs = []
    while True:
        logger.info(f"Getting objects {100 * i} - {100 * (i + 1)}")
        if cursor is not None:
            variables["after"] = cursor
        json = {"query": query, "variables": variables}
        request = HttpRequest(method=HttpMethod.POST, base_url=BASE_URL,
                              endpoint="/graphql", headers=headers, json=json)
        res = send_request(request, tries=3, delay=5).json()
        root = res["data"]
        rate_limit = root["rateLimit"]
        remaining = rate_limit["remaining"]
        reset_at = pd.Timestamp(rate_limit['resetAt'])
        # TODO: sleep if rate limit breached?
        logger.info(
            f"{remaining} units of API quota remaining; resetting at: "
            f"{reset_at}")
        data = get(root, path)
        objs.extend(data["edges"])
        page_info = data["pageInfo"]
        cursor = page_info["endCursor"]
        if not page_info["hasNextPage"]:
            break
        i += 1

    return objs


def get_stargazers(owner: str, name: str, cursor: Optional[str] = None,
                   auth_token: Optional[str] = None):
    objs = _get_objs(STARGAZERS_QUERY,
                     path=("repository", "stargazers"),
                     variables={"owner": owner, "name": name},
                     cursor=cursor,
                     auth_token=auth_token)
    data = [{"login": x["node"]["login"], "starredAt": x["starredAt"],
             "cursor": x["cursor"]} for x in objs]
    data = pd.DataFrame(data)
    if not data.empty:
        data["starredAt"] = pd.to_datetime(
            data["starredAt"]).dt.tz_localize(None)
    return data


def get_pull_requests(owner: str, name: str, cursor: Optional[str] = None,
                      auth_token: Optional[str] = None):
    objs = _get_objs(PULL_REQUESTS_QUERY,
                     path=("repository", "pullRequests"),
                     variables={"owner": owner, "name": name},
                     cursor=cursor,
                     auth_token=auth_token)
    data = [
        {
            "login": _get_login(x["node"]["author"]),
            "createdAt": x["node"]["createdAt"],
            "mergedAt": x["node"]["mergedAt"],
            "additions": x["node"]["additions"],
            "deletions": x["node"]["deletions"],
            "cursor": x["cursor"],
        }
        for x in objs
    ]
    data = pd.DataFrame(data)
    if not data.empty:
        data["mergedAt"] = pd.to_datetime(
            data["mergedAt"]).dt.tz_localize(None)
        data["createdAt"] = pd.to_datetime(
            data["createdAt"]).dt.tz_localize(None)
    return data


def get_repo(owner: str, repo: str,
             auth_token: Optional[str] = None) -> Dict:
    headers = {}
    if auth_token is not None:
        headers = {"Authorization": f"Bearer {auth_token}"}

    request = HttpRequest(method=HttpMethod.GET, base_url=BASE_URL,
                          endpoint=f"/repos/{owner}/{repo}", headers=headers)
    res = send_request(request)
    return res.json()


def get_repo_creation_time(owner: str, repo: str,
                           auth_token: Optional[str] = None):
    repo = get_repo(owner, repo, auth_token)
    return pd.Timestamp(repo["created_at"])
