import logging
from datetime import datetime
from typing import Optional, Dict

import pandas as pd
from pytrade.net.constants import USER_AGENT
from pytrade.net.http import HttpMethod, HttpRequest, _send_request
from pytrade.utils.collections import flatten
from pytrade.utils.retry import retry
import numpy as np

logger = logging.getLogger(__name__)

# set no_silent_downcasting to True so when doing fillna on column with all None
# values the dtype of the column won't change to float
pd.set_option('future.no_silent_downcasting', True)

BASE_URL = "https://www.glassdoor.co.uk"
GRAPH_ENDPOINT = "/graph"
SEARCH_ENDPOINT = "/searchsuggest/typeahead/community"

QUERY = """
  query GetEmployerReviews(
    $applyDefaultCriteria: Boolean, 
    $dynamicProfileId: Int,
    $employerId: Int!,
    $employmentStatuses: [EmploymentStatusEnum],
    $enableKeywordSearch: Boolean!,
    $goc: GOCIdent,
    $isRowProfileEnabled: Boolean,
    $jobTitle: JobTitleIdent,
    $language: String,
    $location: LocationIdent,
    $onlyCurrentEmployees: Boolean,
    $page: Int!,
    $pageSize: Int!,
    $preferredTldId: Int,
    $reviewCategories: [ReviewCategoriesEnum],
    $sort: ReviewsSortOrderEnum,
    $textSearch: String,
    $useRowProfileTldForRatings: Boolean,
    $worldwideFilter: Boolean) {
  employerReviews: employerReviewsRG(
        employerReviewsInput: {
                applyDefaultCriteria: $applyDefaultCriteria, 
                dynamicProfileId: $dynamicProfileId, 
                employer: {id: $employerId},
                employmentStatuses: $employmentStatuses,
                onlyCurrentEmployees: $onlyCurrentEmployees,
                goc: $goc,
                isRowProfileEnabled: $isRowProfileEnabled,
                jobTitle: $jobTitle,
                language: $language,
                location: $location,
                page: {
                    num: $page,
                    size: $pageSize
                },
                preferredTldId: $preferredTldId,
                reviewCategories: $reviewCategories,
                sort: $sort,
                textSearch: $textSearch,
                useRowProfileTldForRatings: $useRowProfileTldForRatings,
                worldwideFilter: $worldwideFilter
            }
      ) {
        allReviewsCount
        currentPage
        filteredReviewsCount
        lastReviewDateTime
        numberOfPages
        ratedReviewsCount
        reviews {
          advice
          cons
          employmentStatus
          featured
          isCurrentJob
          jobTitle {
            text
          }
          languageId
          lengthOfEmployment
          location {
            name
          }
          pros
          ratingBusinessOutlook
          ratingCareerOpportunities
          ratingCeo
          ratingCompensationAndBenefits
          ratingCultureAndValues
          ratingDiversityAndInclusion
          ratingOverall
          ratingRecommendToFriend
          ratingSeniorLeadership
          ratingWorkLifeBalance
          reviewDateTime
          reviewId
          summary
          textSearchHighlightPhrases @include(if: $enableKeywordSearch) {
            field
            phrases {
              length
              position: pos
            }
          }
        }
      }
    }
"""

DEFAULT_VARIABLES = {
    "applyDefaultCriteria": True,
    "employerId": None,
    "employmentStatuses": [],
    "goc": None,
    "jobTitle": None,
    "language": "eng",
    "location": {
        "countryId": None,
        "stateId": None,
        "metroId": None,
        "cityId": None,
    },
    "locationName": "",
    "mlHighlightSearch": None,
    "onlyCurrentEmployees": False,
    "page": 1,
    "preferredTldId": 0,
    "reviewCategories": [],
    "sort": "DATE",
    "textSearch": "",
    "worldwideFilter": False,
    "dynamicProfileId": 1,
    "useRowProfileTldForRatings": False,
    "enableKeywordSearch": False,
}

HEADERS = {
    "Origin": "https://www.glassdoor.co.uk",
    # must include referer header too (since 2024-08-02)
    "Referer": "https://www.glassdoor.co.uk/",
    "Content-Type": "application/json",
    "User-Agent": USER_AGENT,
    # TODO: vary token?
    "Gd-Csrf-Token": "812u392ansdmaklsdl",
}

# use "object" for string column type instead of str so None doesn't get cast
# to literal string "None"
REVIEWS_COLUMN_TYPES = {
    "advice": "object",
    "cons": "object",
    "employmentStatus": "object",  # PART_TIME, FULL_TIME
    "featured": "bool",
    "isCurrentJob": "bool",
    "jobTitle.text": "object",
    "languageId": "object",
    "lengthOfEmployment": int,
    "location.name": "object",
    "pros": "object",
    "ratingBusinessOutlook": "object",  # POSITIVE, NEGATIVE, NEUTRAL
    "ratingCareerOpportunities": float,
    "ratingCeo": "object",  # POSITIVE, NEUTRAL
    "ratingCompensationAndBenefits": float,
    "ratingCultureAndValues": float,
    "ratingDiversityAndInclusion": float,
    "ratingOverall": float,
    "ratingRecommendToFriend": "object",  # POSITIVE, NEGATIVE, NEUTRAL
    "ratingSeniorLeadership": float,
    "ratingWorkLifeBalance": float,
    "reviewId": int,
    "summary": "object"
}


def process_review(review: Dict):
    if review["jobTitle"] is None:
        review["jobTitle"] = {"text": None}
    if review["location"] is None:
        review["location"] = {"name": None}
    return flatten(review)


def get_reviews(employer_id: int, start_time: Optional[datetime] = None,
                page_size: int = 100):
    """
    Gets glassdoor reviews.

    Paramters
    ---------
    employer_id
        Employer ID.
    start_time
        Earliest time to get reviews for.
    page_size
        Page size. API seems to return no data if set to number larger than 100.
    """
    data = []
    variables = DEFAULT_VARIABLES.copy()
    variables["pageSize"] = page_size
    while True:
        logger.info(f"Getting reviews for employer {employer_id}"
                    f" (page {variables['page']})")
        variables["employerId"] = employer_id
        query = [
            {
                "operationName": "GetEmployerReviews",
                "variables": variables,
                "query": QUERY,
            }
        ]
        req = HttpRequest(
            BASE_URL,
            endpoint=GRAPH_ENDPOINT,
            method=HttpMethod.POST,
            json=query,
            headers=HEADERS,
        )
        res = retry(_send_request, args=(req,), initial_interval=30, max_interval=300)
        reviews = res.json()[0]["data"]["employerReviews"]["reviews"]
        if len(reviews):
            reviews = pd.DataFrame([process_review(x) for x in reviews])
            reviews["reviewDateTime"] = pd.to_datetime(reviews["reviewDateTime"],
                                                       format="ISO8601")
            reviews = reviews.set_index("reviewDateTime")
            # fillna below to replace None with np.nan
            reviews = reviews.fillna(np.nan).astype(REVIEWS_COLUMN_TYPES)
            data.append(reviews)
            if start_time is not None and min(reviews.index) <= start_time:
                break
        else:
            break
        variables["page"] += 1
    if len(data):
        return pd.concat(data).sort_index().loc[start_time:]
    data = pd.DataFrame([], columns=list(REVIEWS_COLUMN_TYPES.keys()),
                        index=pd.DatetimeIndex([], name="reviewDateTime"))
    return data.astype(REVIEWS_COLUMN_TYPES)


def search(query: str, num_results: int = 10):
    params = {
        "numSuggestions": num_results,
        "source": "GD_V2",
        "version": "NEW",
        "rf": "full",
        "fallback": "token",
        "input": query
    }
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint=SEARCH_ENDPOINT,
        params=params,
        headers=HEADERS,
    )
    res = _send_request(req)
    res = pd.DataFrame(res.json()["typeahead"])
    res = res.set_index("employerId")
    return res[res["category"] == "company"]


def get_autocomplete_options(query: str, num_results: int = 10):
    headers = {
        "Origin": "https://www.glassdoor.co.uk",
        "User-Agent": USER_AGENT,
        "Gd-Csrf-Token": "abcd",
    }
    req = HttpRequest(
        base_url=BASE_URL,
        endpoint="/api-web/employer/find.htm",
        params={
            "autocomplete": "true",
            "maxEmployersForAutocomplete": str(num_results),
            "term": query
        },
        headers=headers,
    )
    res = _send_request(req)
    return pd.DataFrame(res).set_index("id")
