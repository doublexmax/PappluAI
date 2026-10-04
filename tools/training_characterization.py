from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys
from typing import Any


MODULES = {
    "baseline": {
        "environment": "src.environment",
        "model": "src.model",
        "core": "src.training_core",
        "curriculum": "src.long_train",
    },
    "refactor": {
        "environment": "src.game.environment",
        "model": "src.model.network",
        "core": "src.training.core",
        "curriculum": "src.training.curriculum",
    },
}


def normalize(value: Any) -> Any:
    if hasattr(value, "detach") and hasattr(value, "dtype"):
        tensor = value.detach().cpu().contiguous()
        return {
            "tensor_dtype": str(tensor.dtype),
            "tensor_shape": list(tensor.shape),
            "tensor_sha256": hashlib.sha256(
                bytes(
                    tensor.reshape(-1)
                    .view(sys.modules["torch"].uint8)
                    .tolist()
                )
            ).hexdigest(),
        }
    if isinstance(value, dict):
        return {
            str(key): normalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [normalize(item) for item in value]
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError("unsupported characterization value %r" % (type(value),))


def digest(value: Any) -> str:
    encoded = json.dumps(
        normalize(value),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def module_set(layout: str) -> dict[str, Any]:
    return {
        name: importlib.import_module(path)
        for name, path in MODULES[layout].items()
    }


def game_signature(environment: Any) -> dict[str, Any]:
    config = environment.GameConfig(
        num_decks=2,
        cards_in_hand=3,
        required_sequences=1,
        max_turns=5,
        recycle_discard=True,
    )
    env = environment.PappluEnv(config)
    observation = env.reset(seed=24680)
    rng = random.Random(13579)
    steps = []
    for _ in range(8):
        encoded = environment.encode_observation(observation)
        mask = environment.legal_action_mask(observation)
        legal = [index for index, allowed in enumerate(mask) if allowed]
        action = rng.choice(legal)
        steps.append(
            {
                "phase": observation.phase.value,
                "encoded": digest(encoded),
                "legal": legal,
                "action": action,
            }
        )
        observation = env.step(action)
        if getattr(observation, "done", False):
            break
    return {
        "config": config.to_dict(),
        "state_dim": environment.STATE_DIM,
        "num_actions": environment.NUM_ACTIONS,
        "encoding_version": environment.ENCODING_VERSION,
        "steps": steps,
        "terminal": bool(getattr(observation, "done", False)),
    }


def prediction_signature(torch: Any, network: Any, encoded: Any) -> list[str]:
    state = torch.tensor(
        encoded,
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
        values = network(state)[0].detach().cpu().tolist()
    return [float(value).hex() for value in values]


def loaded_checkpoint_signature(
    torch: Any,
    environment: Any,
    model: Any,
    path: Path,
) -> dict[str, Any]:
    network, config, meta = model.load_checkpoint(str(path))
    observation = environment.PappluEnv(config).reset(seed=97531)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "checkpoint_version": raw["checkpoint_version"],
        "encoding_version": raw["encoding_version"],
        "state_dim": raw["state_dim"],
        "num_actions": raw["num_actions"],
        "architecture": network.architecture,
        "game_config": config.to_dict(),
        "meta": meta,
        "model_state": digest(network.state_dict()),
        "prediction": prediction_signature(
            torch,
            network,
            environment.encode_observation(observation),
        ),
    }


def create_checkpoints(
    torch: Any,
    environment: Any,
    model: Any,
    output: Path,
) -> dict[str, Any]:
    torch.manual_seed(20250301)
    config = environment.GameConfig(
        num_decks=2,
        cards_in_hand=3,
        required_sequences=1,
        max_turns=5,
        recycle_discard=True,
    )
    network = model.QNetwork(architecture="mlp")
    v2_path = output / "model-v2.pt"
    model.save_checkpoint(
        str(v2_path),
        network,
        config,
        meta={"purpose": "package-characterization"},
    )
    v1_payload = model.build_checkpoint(
        network,
        config,
        meta={"purpose": "package-characterization"},
    )
    v1_payload["checkpoint_version"] = 1
    del v1_payload["architecture"]
    del v1_payload["game_config"]["recycle_discard"]
    v1_path = output / "model-v1.pt"
    torch.save(v1_payload, v1_path)
    return {
        "v1": loaded_checkpoint_signature(
            torch,
            environment,
            model,
            v1_path,
        ),
        "v2": loaded_checkpoint_signature(
            torch,
            environment,
            model,
            v2_path,
        ),
    }


def replay_signature(environment: Any, core: Any) -> dict[str, Any]:
    replay = core.EpisodeReplay(capacity=8)

    def transition(index: int) -> tuple[tuple[float, ...], int, float]:
        state = (float(index),) + (0.0,) * (environment.STATE_DIM - 1)
        return state, index, float(index) / 10.0

    replay.append(tuple(transition(index) for index in range(3)))
    replay.append(tuple(transition(index) for index in range(3, 7)))
    sampled = replay.sample(random.Random(86420), 12)
    return {
        "state_version": replay.STATE_VERSION,
        "length": len(replay),
        "state": digest(replay.state_dict()),
        "sample": digest(sampled),
        "sample_actions": [item[1] for item in sampled],
        "discounted_returns": core.discounted_returns(
            [0.0, 0.25, 1.0],
            0.5,
        ),
    }


def random_future(state: Any) -> list[str]:
    rng = random.Random()
    rng.setstate(state)
    return [rng.random().hex() for _ in range(5)]


def torch_future(torch: Any, state: Any) -> list[str]:
    before = torch.get_rng_state()
    try:
        torch.set_rng_state(state)
        return [
            float(value).hex()
            for value in torch.rand(5, dtype=torch.float32).tolist()
        ]
    finally:
        torch.set_rng_state(before)


def session_signature(
    torch: Any,
    environment: Any,
    model: Any,
    session: Any,
) -> dict[str, Any]:
    state = session.state_dict()
    observation = environment.PappluEnv(
        session.config.game_config
    ).reset(seed=112233)
    action_rng = random.Random()
    action_rng.setstate(state["rng_states"]["action"])
    next_action = model.select_action(
        session.network,
        observation,
        epsilon=session.current_epsilon,
        rng=action_rng,
    )
    replay_rng = random.Random()
    replay_rng.setstate(state["rng_states"]["replay"])
    replay_sample = (
        session.replay.sample(replay_rng, 8)
        if len(session.replay)
        else ()
    )
    return {
        "versions": {
            "training": state["training_state_version"],
            "encoding": state["encoding_version"],
            "model": state["model_checkpoint_version"],
            "algorithm": state["algorithm"],
        },
        "training_config": state["training_config"],
        "initial_seed": state["initial_seed"],
        "counters": state["counters"],
        "curriculum": state["curriculum"],
        "validation": state["validation"],
        "source_trace": state["source_trace"],
        "network_state": digest(state["network_state_dict"]),
        "optimizer_state": digest(state["optimizer_state_dict"]),
        "replay_state": digest(state["replay_state_dict"]),
        "baseline_model_state": digest(
            state["baseline_model_state_dict"]
        ),
        "best_model_state": digest(state["best_model_state_dict"]),
        "rng_states": {
            name: digest(value)
            for name, value in state["rng_states"].items()
        },
        "rng_future": {
            "deal": random_future(state["rng_states"]["deal"]),
            "action": random_future(state["rng_states"]["action"]),
            "replay": random_future(state["rng_states"]["replay"]),
            "torch": torch_future(torch, state["rng_states"]["torch"]),
        },
        "next_action": next_action,
        "replay_sample": digest(replay_sample),
        "prediction": prediction_signature(
            torch,
            session.network,
            environment.encode_observation(observation),
        ),
    }


def create_session_characterization(
    torch: Any,
    modules: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    environment = modules["environment"]
    curriculum = modules["curriculum"]
    config = curriculum.TrainingConfig(
        max_turns=1,
        replay_capacity=32,
        batch_size=2,
        updates_per_episode=1,
        validation_games=1,
        validate_every=100,
        minimum_stage_episodes=1,
    )
    session = curriculum.TrainingSession(config, seed=41)
    session.stage_index = len(config.curriculum_distances)
    first_episode = session.run_episode()
    saved = session_signature(
        torch,
        environment,
        modules["model"],
        session,
    )
    state_dir = output / "training-state"
    curriculum.save_session_checkpoint(session, state_dir)
    resumed = curriculum.TrainingSession.load(
        str(state_dir / "latest-state.pt"),
        config,
    )
    loaded = session_signature(
        torch,
        environment,
        modules["model"],
        resumed,
    )
    second_episode = resumed.run_episode()
    continued = session_signature(
        torch,
        environment,
        modules["model"],
        resumed,
    )
    return {
        "first_episode": first_episode,
        "saved": saved,
        "loaded": loaded,
        "second_episode": second_episode,
        "continued": continued,
    }


def cross_load_baseline(
    torch: Any,
    modules: dict[str, Any],
    baseline_output: Path,
) -> dict[str, Any]:
    environment = modules["environment"]
    model = modules["model"]
    curriculum = modules["curriculum"]
    checkpoints = {
        version: loaded_checkpoint_signature(
            torch,
            environment,
            model,
            baseline_output / ("model-%s.pt" % version),
        )
        for version in ("v1", "v2")
    }
    config = curriculum.TrainingConfig(
        max_turns=1,
        replay_capacity=32,
        batch_size=2,
        updates_per_episode=1,
        validation_games=1,
        validate_every=100,
        minimum_stage_episodes=1,
    )
    session = curriculum.TrainingSession.load(
        str(baseline_output / "training-state" / "latest-state.pt"),
        config,
    )
    loaded = session_signature(
        torch,
        environment,
        model,
        session,
    )
    second_episode = session.run_episode()
    continued = session_signature(
        torch,
        environment,
        model,
        session,
    )
    return {
        "checkpoints": checkpoints,
        "session": {
            "loaded": loaded,
            "second_episode": second_episode,
            "continued": continued,
        },
    }


def difference(left: Any, right: Any, path: str = "$") -> str | None:
    if type(left) is not type(right):
        return "%s type %s != %s" % (
            path,
            type(left).__name__,
            type(right).__name__,
        )
    if isinstance(left, dict):
        if set(left) != set(right):
            return "%s keys %r != %r" % (
                path,
                sorted(left),
                sorted(right),
            )
        for key in sorted(left):
            found = difference(left[key], right[key], path + "." + key)
            if found:
                return found
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return "%s length %d != %d" % (path, len(left), len(right))
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            found = difference(
                left_item,
                right_item,
                "%s[%d]" % (path, index),
            )
            if found:
                return found
        return None
    if left != right:
        return "%s %r != %r" % (path, left, right)
    return None


def assert_equal(left: Any, right: Any, label: str) -> None:
    found = difference(left, right)
    if found:
        raise AssertionError("%s differs: %s" % (label, found))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        choices=tuple(MODULES),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-output", type=Path)
    args = parser.parse_args()
    if args.layout == "refactor" and args.baseline_output is None:
        parser.error("--baseline-output is required for refactor")

    source_root = args.source_root.resolve()
    tool_root = Path(__file__).resolve().parents[1]
    remaining_paths = []
    for entry in sys.path:
        resolved = Path(entry or Path.cwd()).resolve()
        if resolved != tool_root and resolved != source_root:
            remaining_paths.append(entry)
    sys.path[:] = [str(source_root), *remaining_paths]
    args.output.mkdir(parents=True, exist_ok=True)
    modules = module_set(args.layout)
    torch = importlib.import_module("torch")
    torch.set_num_threads(1)
    result = {
        "game": game_signature(modules["environment"]),
        "checkpoints": create_checkpoints(
            torch,
            modules["environment"],
            modules["model"],
            args.output,
        ),
        "replay": replay_signature(
            modules["environment"],
            modules["core"],
        ),
        "session": create_session_characterization(
            torch,
            modules,
            args.output,
        ),
    }
    normalized = normalize(result)

    if args.layout == "refactor":
        baseline_result = json.loads(
            (args.baseline_output / "result.json").read_text(
                encoding="utf-8"
            )
        )
        assert_equal(
            baseline_result,
            normalized,
            "baseline and refactor characterization",
        )
        cross = normalize(
            cross_load_baseline(
                torch,
                modules,
                args.baseline_output,
            )
        )
        assert_equal(
            baseline_result["checkpoints"],
            cross["checkpoints"],
            "baseline checkpoints loaded by refactor",
        )
        assert_equal(
            baseline_result["session"]["loaded"],
            cross["session"]["loaded"],
            "baseline training state loaded by refactor",
        )
        assert_equal(
            baseline_result["session"]["second_episode"],
            cross["session"]["second_episode"],
            "baseline resume episode loaded by refactor",
        )
        assert_equal(
            baseline_result["session"]["continued"],
            cross["session"]["continued"],
            "baseline resume update loaded by refactor",
        )

    (args.output / "result.json").write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        "%s characterization matched protected behavior"
        % args.layout
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
