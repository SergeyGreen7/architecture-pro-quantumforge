from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import chromadb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from update_index import EXPECTED_DIMENSION, UpdateConfig, update_index


class FakeEmbeddingModel:
    def get_embedding_dimension(self) -> int:
        return EXPECTED_DIMENSION

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        embeddings = np.zeros((len(texts), EXPECTED_DIMENSION), dtype=np.float32)
        for row, text in enumerate(texts):
            embeddings[row, hash(text) % EXPECTED_DIMENSION] = 1.0
        return embeddings


class IncrementalUpdateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.source = root / "source"
        self.source.mkdir()
        self.index = root / "index"
        self.state = root / "state.json"
        self.log = root / "update.jsonl"
        self.lock = root / "update.lock"
        self.cache = root / "cache"
        self.config = UpdateConfig(
            source_dir=self.source,
            index_dir=self.index,
            state_path=self.state,
            log_path=self.log,
            lock_path=self.lock,
            model_cache_dir=self.cache,
        )
        (self.source / "alpha.txt").write_text(
            "Alpha\n=====\n\nFirst version.", encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def model_factory(_: UpdateConfig) -> FakeEmbeddingModel:
        return FakeEmbeddingModel()

    def collection_count(self) -> int:
        client = chromadb.PersistentClient(path=str(self.index))
        return client.get_collection("quantumforge_knowledge_base").count()

    def test_add_change_delete_and_noop(self) -> None:
        first = update_index(self.config, model_factory=self.model_factory)
        self.assertEqual(first["new_documents"], 1)
        self.assertEqual(first["new_chunks"], 1)
        self.assertEqual(first["final_index_size"], 1)

        def unexpected_model(_: UpdateConfig) -> FakeEmbeddingModel:
            raise AssertionError("model must not load for a no-op update")

        noop = update_index(self.config, model_factory=unexpected_model)
        self.assertFalse(noop["changes_detected"])
        self.assertEqual(noop["embeddings_generated"], 0)

        (self.source / "alpha.txt").write_text(
            "Alpha\n=====\n\nChanged version.", encoding="utf-8"
        )
        changed = update_index(self.config, model_factory=self.model_factory)
        self.assertEqual(changed["changed_documents"], 1)
        self.assertEqual(changed["updated_chunks"], 1)
        self.assertEqual(changed["removed_chunks"], 1)
        self.assertEqual(self.collection_count(), 1)

        (self.source / "beta.txt").write_text(
            "Beta\n====\n\nNew daily document.", encoding="utf-8"
        )
        added = update_index(self.config, model_factory=self.model_factory)
        self.assertEqual(added["new_documents"], 1)
        self.assertEqual(added["new_chunks"], 1)
        self.assertEqual(self.collection_count(), 2)

        (self.source / "alpha.txt").unlink()
        deleted = update_index(self.config, model_factory=unexpected_model)
        self.assertEqual(deleted["deleted_documents"], 1)
        self.assertEqual(deleted["removed_chunks"], 1)
        self.assertEqual(self.collection_count(), 1)

        state = json.loads(self.state.read_text(encoding="utf-8"))
        self.assertEqual(state["index_chunk_count"], 1)
        self.assertEqual(len(state["documents"]), 1)


if __name__ == "__main__":
    unittest.main()
