from typing import Optional

from pydantic import BaseModel, validator
import pubmed_parser as pp

# from linking.utils.preprocessing_regex import pubmed_process_title
from .preprocessing_regex import pubmed_process_title


class Article(BaseModel):
    title: Optional[str] = None
    title_normalized: Optional[str] = None
    pmid: str
    issue: Optional[str] = None
    pages: Optional[str] = None
    abstract: Optional[str] = None
    fulltext: Optional[str] = None
    fulltext_license: Optional[str] = None
    journal: Optional[str] = None
    authors: Optional[str] = None
    pubdate: Optional[str] = None
    doi: Optional[str] = None
    affiliations: Optional[str] = None
    medline_ta: Optional[str] = None
    nlm_unique_id: Optional[str] = None
    issn_linking: Optional[str] = None
    country: Optional[str] = None
    mesh_terms: Optional[str] = None
    publication_types: Optional[str] = None
    chemical_list: Optional[str] = None
    keywords: Optional[str] = None
    references: Optional[str] = None
    delete: Optional[bool] = False
    pmc: Optional[str] = None
    other_id: Optional[str] = None

    @validator("title")
    def strip_brackets(cls, value):
        if value:
            value = value.strip("[]")
            if value.endswith("]."):
                value = value[:-2] + "."
        return value

    @validator("title_normalized", always=True)
    def process_title(cls, v, values):
        # do nothing if we parsed a normalized title
        if v:
            return v
        # try to create a normalized title
        title = values.get("title")
        normalized = pubmed_process_title(title)
        if normalized != "":
            return normalized
        else:
            return None


def enforce_none(dict):
    for k, v in dict.items():
        if v == "":
            dict[k] = None
    return dict


def article_from_file(file):
    articles = pp.parse_medline_xml(
        file, year_info_only=False, nlm_category=True, author_list=False
    )
    articles = [enforce_none(a) for a in articles]
    return [Article.parse_obj(a) for a in articles]


if __name__ == "__main__":
    file = "./linking/pubmed/ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n1114.xml"
    articles = article_from_file(file)
    print(articles[0])
