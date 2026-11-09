from types import SimpleNamespace

from workflow import CorrectiveRAGWorkflow


class FakeLLM:
    """Grades chunks containing 'relevant' as yes; echoes for other prompts."""

    async def acomplete(self, prompt: str):
        if prompt.startswith("You grade"):
            chunk = prompt.split("Chunk:\n")[1]
            return SimpleNamespace(text="yes" if "relevant" in chunk.split("\n")[0] else "no")
        if prompt.startswith("Rewrite"):
            return SimpleNamespace(text="rewritten query")
        return SimpleNamespace(text=f"ANSWER<{prompt}>")


class FakeIndex:
    def __init__(self, chunks):
        self.chunks = chunks

    def as_retriever(self, **_):
        nodes = [SimpleNamespace(get_content=lambda c=c: c) for c in self.chunks]
        return SimpleNamespace(retrieve=lambda _query: nodes)


class FakeFirecrawl:
    def __init__(self, fail=False):
        self.fail = fail
        self.queries = []

    async def search(self, query, **_):
        self.queries.append(query)
        if self.fail:
            raise RuntimeError("boom")
        page = SimpleNamespace(
            markdown="web page text", metadata=SimpleNamespace(url="https://example.com", title="Example")
        )
        return SimpleNamespace(web=[page])


def make_workflow(chunks, firecrawl):
    index = FakeIndex(chunks) if chunks is not None else None
    wf = CorrectiveRAGWorkflow(index=index, firecrawl_api_key="fc-test", llm=FakeLLM(), timeout=10)
    wf.firecrawl = firecrawl
    return wf


async def test_relevant_documents_skip_web_search():
    firecrawl = FakeFirecrawl()
    result = await make_workflow(["relevant a", "relevant b"], firecrawl).run(query_str="q")
    assert firecrawl.queries == []
    assert result["doc_chunks_used"] == 2
    assert "relevant a" in result["answer"]


async def test_irrelevant_chunk_is_filtered_and_triggers_web_search():
    firecrawl = FakeFirecrawl()
    result = await make_workflow(["relevant a", "noise"], firecrawl).run(query_str="q")
    assert firecrawl.queries == ["rewritten query"]
    assert result["sources"] == [{"title": "Example", "url": "https://example.com"}]
    assert "noise" not in result["answer"]
    assert "web page text" in result["answer"]


async def test_no_index_answers_from_web():
    result = await make_workflow(None, FakeFirecrawl()).run(query_str="q")
    assert result["doc_chunks_used"] == 0
    assert "web page text" in result["answer"]


async def test_web_failure_is_reported_not_raised():
    result = await make_workflow(["noise"], FakeFirecrawl(fail=True)).run(query_str="q")
    assert result["web_error"] == "RuntimeError: boom"
    assert "could not find" in result["answer"]
