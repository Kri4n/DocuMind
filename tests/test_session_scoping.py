import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from qdrant_client.models import Range

from custom_types import RAGChunkAndSrc
from vector_db import QdrantStorage
import main


class QdrantSessionScopingTests(unittest.TestCase):
    def setUp(self):
        self.query_args = {}

        def query_points(**kwargs):
            self.query_args.update(kwargs)
            return SimpleNamespace(
                points=[
                    SimpleNamespace(
                        payload={"text": "session text", "source": "doc.pdf"}
                    )
                ]
            )

        self.store = QdrantStorage.__new__(QdrantStorage)
        self.store.collection = "docs"
        self.store.client = SimpleNamespace(query_points=query_points)

    def test_search_filters_by_session(self):
        result = self.store.search([0.1, 0.2], "session-a")

        self.assertEqual(result["contexts"], ["session text"])
        condition = self.query_args["query_filter"].must[0]
        self.assertEqual(condition.key, "session_id")
        self.assertEqual(condition.match.value, "session-a")

    def test_delete_expired_sessions_filters_by_expiration(self):
        captured = {}
        self.store.client.delete = lambda **kwargs: captured.update(kwargs)

        self.store.delete_expired_sessions(now=1234)

        selector = captured["points_selector"]
        condition = selector.filter.must[0]
        self.assertEqual(condition.key, "session_expires_at")
        self.assertEqual(condition.range, Range(lte=1234))

    def test_touch_session_extends_only_that_sessions_expiry(self):
        captured = {}
        self.store.client.set_payload = lambda **kwargs: captured.update(kwargs)

        with patch("vector_db.time.time", return_value=1000):
            self.store.touch_session("session-a", ttl_seconds=300)

        self.assertEqual(captured["payload"], {"session_expires_at": 1300})
        condition = captured["points"].filter.must[0]
        self.assertEqual(condition.key, "session_id")
        self.assertEqual(condition.match.value, "session-a")

    def test_ensure_session_indexes_creates_missing_indexes_once(self):
        created = []

        def get_collection(collection):
            self.assertEqual(collection, "docs")
            return SimpleNamespace(payload_schema={"session_id": object()})

        self.store.client.get_collection = get_collection
        self.store.client.create_payload_index = lambda **kwargs: created.append(kwargs)

        self.store.ensure_session_indexes()

        self.assertEqual(len(created), 1)
        self.assertEqual(created[0]["field_name"], "session_expires_at")


class IngestionSessionIsolationTests(unittest.TestCase):
    def test_same_document_in_different_sessions_gets_distinct_point_ids(self):
        upserts = []
        touched_sessions = []

        class FakeStore:
            def __init__(self, collection, dim):
                self.collection = collection
                self.dim = dim

            def upsert(self, ids, vectors, payloads):
                upserts.append((ids, vectors, payloads))

            def touch_session(self, session_id, ttl_seconds):
                touched_sessions.append((session_id, ttl_seconds))

        with (
            patch("main.embed_texts", return_value=[[0.1, 0.2]]),
            patch("main.QdrantStorage", FakeStore),
        ):
            from main import _upsert_pdf

            _upsert_pdf(
                RAGChunkAndSrc(
                    chunks=["text"],
                    source_id="document.pdf",
                    session_id="session-a",
                )
            )
            _upsert_pdf(
                RAGChunkAndSrc(
                    chunks=["text"],
                    source_id="document.pdf",
                    session_id="session-b",
                )
            )

        self.assertNotEqual(upserts[0][0], upserts[1][0])
        self.assertEqual(upserts[0][2][0]["session_id"], "session-a")
        self.assertEqual(upserts[1][2][0]["session_id"], "session-b")
        self.assertEqual(touched_sessions, [])

    def test_query_route_passes_only_the_requested_session(self):
        session_id = "7e12a779-7e78-4e31-bd37-f3f99dc23027"
        request = main.QueryRequest(question="question", session_id=session_id)
        background_tasks = main.BackgroundTasks()
        expected = main.RAGQueryResult(answer="answer", sources=[], num_contexts=0)

        with patch("main._answer_question", return_value=expected) as answer:
            result = asyncio.run(main.query_documents(request, background_tasks))

        self.assertEqual(result, expected)
        answer.assert_called_once_with("question", 5, session_id)
        self.assertEqual(len(background_tasks.tasks), 1)
        self.assertEqual(background_tasks.tasks[0].func, main._touch_session)
        self.assertEqual(background_tasks.tasks[0].args, (session_id,))

    def test_cleanup_waits_before_first_sweep(self):
        async def stop_after_initial_wait(seconds):
            self.assertEqual(seconds, main.SESSION_CLEANUP_INTERVAL_SECONDS)
            raise asyncio.CancelledError

        cleanup = AsyncMock()
        with (
            patch("main.asyncio.sleep", side_effect=stop_after_initial_wait),
            patch("main._cleanup_expired_sessions", cleanup),
            self.assertRaises(asyncio.CancelledError),
        ):
            asyncio.run(main._session_cleanup_loop())

        cleanup.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
