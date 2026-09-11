from __future__ import annotations

import httpx

from multiagent.tools.ncbi import NCBIClient


def test_temporal_query_contains_cutoff():
    query = NCBIClient.temporal_query("migraine magnesium", 1985)
    assert query == "(migraine magnesium) AND 1800:1985[dp]"


def test_pubmed_search_and_fetch_parse_structured_article():
    seen_terms: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("esearch.fcgi"):
            seen_terms.append(request.url.params["term"])
            return httpx.Response(
                200,
                json={"esearchresult": {"idlist": ["12345"], "count": "1"}},
            )

        if request.url.path.endswith("efetch.fcgi"):
            xml = """
            <PubmedArticleSet>
              <PubmedArticle>
                <MedlineCitation>
                  <PMID>12345</PMID>
                  <Article>
                    <Journal>
                      <JournalIssue><PubDate><Year>1982</Year></PubDate></JournalIssue>
                      <Title>Mock Journal</Title>
                    </Journal>
                    <ArticleTitle>Magnesium and <i>migraine</i></ArticleTitle>
                    <Abstract><AbstractText>Mock abstract.</AbstractText></Abstract>
                  </Article>
                  <MeshHeadingList>
                    <MeshHeading><DescriptorName>Magnesium</DescriptorName></MeshHeading>
                    <MeshHeading><DescriptorName>Migraine Disorders</DescriptorName></MeshHeading>
                  </MeshHeadingList>
                </MedlineCitation>
              </PubmedArticle>
            </PubmedArticleSet>
            """
            return httpx.Response(200, text=xml)

        raise AssertionError(f"Unexpected request: {request.url}")

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NCBIClient(http_client=http_client)
    articles = client.search_pubmed("migraine magnesium", before_year=1985, top_k=3)

    assert seen_terms == ["(migraine magnesium) AND 1800:1985[dp]"]
    assert len(articles) == 1
    assert articles[0].pmid == "12345"
    assert articles[0].title == "Magnesium and migraine"
    assert articles[0].year == 1982
    assert articles[0].journal == "Mock Journal"
    assert articles[0].mesh_terms == ("Magnesium", "Migraine Disorders")


def test_pair_mention_count_is_temporally_constrained():
    captured_term = ""

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_term
        captured_term = request.url.params["term"]
        return httpx.Response(
            200,
            json={"esearchresult": {"idlist": [], "count": "7"}},
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = NCBIClient(http_client=http_client)

    count = client.pair_mention_count("Migraine", "Magnesium", 1985)

    assert count == 7
    assert '"Migraine" AND "Magnesium"' in captured_term
    assert "1800:1985[dp]" in captured_term
