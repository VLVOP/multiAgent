from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

EUTILS_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


@dataclass(frozen=True)
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    year: int | None
    journal: str | None
    mesh_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MeshConcept:
    uid: str
    name: str
    mesh_ui: str | None
    scope_note: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(part.strip() for part in element.itertext() if part.strip())


def _year_from_article(article: ET.Element) -> int | None:
    year = article.findtext(".//JournalIssue/PubDate/Year")
    if year and year.isdigit():
        return int(year)

    medline_date = article.findtext(".//JournalIssue/PubDate/MedlineDate") or ""
    match = re.search(r"\b(18|19|20)\d{2}\b", medline_date)
    return int(match.group(0)) if match else None


class NCBIClient:
    """Small NCBI E-utilities client with temporal constraints built into PubMed search."""

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        timeout: float = 20.0,
    ) -> None:
        self._owns_client = http_client is None
        self.http = http_client or httpx.Client(timeout=timeout, follow_redirects=True)

    def close(self) -> None:
        if self._owns_client:
            self.http.close()

    def __enter__(self) -> "NCBIClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _common_params(self) -> dict[str, str]:
        params = {"tool": os.getenv("NCBI_TOOL", "arrowsmith_multiagent")}
        email = os.getenv("NCBI_EMAIL")
        api_key = os.getenv("NCBI_API_KEY")
        if email:
            params["email"] = email
        if api_key:
            params["api_key"] = api_key
        return params

    def _get(self, endpoint: str, **params: Any) -> httpx.Response:
        merged = {**self._common_params(), **params}
        response = self.http.get(f"{EUTILS_BASE_URL}/{endpoint}", params=merged)
        response.raise_for_status()
        return response

    @staticmethod
    def temporal_query(query: str, before_year: int) -> str:
        if before_year < 1800 or before_year > 2100:
            raise ValueError("before_year must be between 1800 and 2100")
        return f"({query}) AND 1800:{before_year}[dp]"

    def search_pubmed_ids(self, query: str, before_year: int, top_k: int = 10) -> list[str]:
        response = self._get(
            "esearch.fcgi",
            db="pubmed",
            term=self.temporal_query(query, before_year),
            retmode="json",
            retmax=max(1, min(top_k, 100)),
            sort="relevance",
        )
        payload = response.json()
        return list(payload.get("esearchresult", {}).get("idlist", []))

    def count_pubmed(self, query: str, before_year: int) -> int:
        response = self._get(
            "esearch.fcgi",
            db="pubmed",
            term=self.temporal_query(query, before_year),
            retmode="json",
            retmax=0,
        )
        count = response.json().get("esearchresult", {}).get("count", "0")
        return int(count)

    def fetch_pubmed_articles(self, pmids: list[str]) -> list[PubMedArticle]:
        if not pmids:
            return []

        response = self._get(
            "efetch.fcgi",
            db="pubmed",
            id=",".join(pmids),
            retmode="xml",
        )
        root = ET.fromstring(response.text)
        articles: list[PubMedArticle] = []

        for pubmed_article in root.findall(".//PubmedArticle"):
            citation = pubmed_article.find("MedlineCitation")
            article = citation.find("Article") if citation is not None else None
            if citation is None or article is None:
                continue

            pmid = citation.findtext("PMID") or ""
            title = _text(article.find("ArticleTitle"))
            abstract_parts = [_text(node) for node in article.findall("Abstract/AbstractText")]
            abstract = "\n".join(part for part in abstract_parts if part)
            journal = _text(article.find("Journal/Title")) or None
            mesh_terms = tuple(
                _text(node)
                for node in citation.findall("MeshHeadingList/MeshHeading/DescriptorName")
                if _text(node)
            )

            articles.append(
                PubMedArticle(
                    pmid=pmid,
                    title=title,
                    abstract=abstract,
                    year=_year_from_article(article),
                    journal=journal,
                    mesh_terms=mesh_terms,
                )
            )

        return articles

    def search_pubmed(
        self,
        query: str,
        before_year: int,
        top_k: int = 10,
    ) -> list[PubMedArticle]:
        pmids = self.search_pubmed_ids(query, before_year, top_k)
        return self.fetch_pubmed_articles(pmids)

    def pair_evidence(
        self,
        entity_a: str,
        entity_b: str,
        before_year: int,
        top_k: int = 5,
    ) -> list[PubMedArticle]:
        """Retrieve pre-cutoff papers mentioning both entities.

        This is candidate evidence, not proof of a typed/causal relation.
        """
        query = f'"{entity_a}" AND "{entity_b}"'
        return self.search_pubmed(query, before_year, top_k)

    def pair_mention_count(self, entity_a: str, entity_b: str, before_year: int) -> int:
        query = f'"{entity_a}" AND "{entity_b}"'
        return self.count_pubmed(query, before_year)

    def search_mesh(self, term: str, top_k: int = 10) -> list[MeshConcept]:
        search_response = self._get(
            "esearch.fcgi",
            db="mesh",
            term=term,
            retmode="json",
            retmax=max(1, min(top_k, 50)),
        )
        uids = list(search_response.json().get("esearchresult", {}).get("idlist", []))
        if not uids:
            return []

        summary_response = self._get(
            "esummary.fcgi",
            db="mesh",
            id=",".join(uids),
            retmode="json",
        )
        result = summary_response.json().get("result", {})

        concepts: list[MeshConcept] = []
        for uid in uids:
            item = result.get(uid, {})
            mesh_terms = item.get("ds_meshterms") or []
            name = (
                item.get("title")
                or item.get("ds_meshheading")
                or (mesh_terms[0] if mesh_terms else None)
                or uid
            )
            concepts.append(
                MeshConcept(
                    uid=uid,
                    name=str(name),
                    mesh_ui=item.get("ds_meshui"),
                    scope_note=item.get("ds_scopenote"),
                )
            )
        return concepts
