from pydantic import BaseModel, validator
from typing import List

from . import Report, Article

# from linking.utils.report import Report
# from linking.utils.article import Article


class Match(BaseModel):
    article: Article
    reports: List[Report]

    @validator("reports")
    def reports_not_empty(cls, v):
        if len(v) == 0:
            raise ValueError("Can not initialize match with no reports.")
        return v
