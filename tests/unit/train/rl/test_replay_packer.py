import gzip
import json

from prime_rl.trainer.rl.packer import load_replay_batch


def test_load_replay_batch(tmp_path):
    records = [
        {
            "sample_id": "sample-1",
            "rollout_id": "rollout-1",
            "task_id": "task-1",
            "input_ids": [1, 2, 3],
            "loss_mask": [False, True, True],
            "advantage": -1.0,
            "rollout_logprobs": [0.0, -0.1, -0.2],
        },
        {
            "sample_id": "sample-2",
            "rollout_id": "rollout-2",
            "task_id": "task-1",
            "input_ids": [4, 5],
            "loss_mask": [False, True],
            "advantage": 1.0,
            "rollout_logprobs": [0.0, -0.3],
        },
    ]
    manifest = {
        "version": 1,
        "artifact_id": "prepared",
        "synthetic_artifact_id": "synthetic",
        "rollout_count": 2,
        "task_count": 1,
        "sample_count": 2,
        "training_tokens": 5,
        "loss_tokens": 3,
        "preparer_revision": "v2",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with gzip.open(tmp_path / "prime.jsonl.gz", "wt") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")

    samples = load_replay_batch(tmp_path)

    assert [sample.token_ids for sample in samples] == [[1, 2, 3], [4, 5]]
    assert samples[0].advantages == [-1.0, -1.0, -1.0]
    assert samples[1].mask == [False, True]
