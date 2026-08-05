"""A policy behind a wire: what it accepts, and how it fails.

Most of this runs against an injected transport, because a fake socket tests
nothing that a fake function does not. One test goes over a real loopback HTTP
server, because the default transport is the part a user actually gets and
"works with the standard library" is a claim worth checking rather than
asserting.
"""

from __future__ import annotations

import json
import threading
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest
from gantry_policy_served import Endpoint, ServedPolicy, decode_action, encode_observation

from gantry.conformance import check_policy
from gantry.contracts.policy import EpisodeContext, Observation
from gantry.errors import ComponentError, ConfigError
from gantry.spine import ChannelSpec

ACTION = ChannelSpec("action", "vector", (3,), "float32", units="m", semantics="actuation")
STATE = ChannelSpec("state", "vector", (3,), "float32", units="m", semantics="position")

URL = "http://policy.invalid/act"


def transport_returning(*responses, calls: list | None = None):
    """A transport that answers with each response in turn, then repeats the last."""
    payloads = list(responses)

    def transport(url, body, headers, timeout):
        if calls is not None:
            calls.append({"url": url, "body": json.loads(body), "headers": dict(headers)})
        payload = payloads.pop(0) if len(payloads) > 1 else payloads[0]
        if isinstance(payload, Exception):
            raise payload
        return json.dumps(payload).encode()

    return transport


def policy(*responses, calls=None, **kwargs) -> ServedPolicy:
    kwargs.setdefault("chunk", 4)
    kwargs.setdefault("deterministic", True)
    return ServedPolicy(
        URL, ACTION, [STATE], transport=transport_returning(*responses, calls=calls), **kwargs
    )


def observation(step: int = 0) -> Observation:
    return Observation(step, {"state": np.zeros(3, dtype="float32")})


# -- the contract ----------------------------------------------------------


def test_conforms():
    served = policy({"action": [[0.1, 0.2, 0.3]]})
    verdict = check_policy(served, [observation(0), observation(1)], strict=True)
    assert verdict.ok, verdict.explain()


def test_it_declares_the_endpoint_and_admits_it_may_be_stateful():
    descriptor = policy({"action": [[0.0, 0.0, 0.0]]}).descriptor()
    assert descriptor.metadata["endpoint"] == URL
    assert descriptor.provides["stateful"] is True
    # The client is what runs here; the model is already somewhere else.
    assert descriptor.isolation == "in-process"


def test_the_declared_observation_requirement_is_what_it_says():
    requirement = policy({"action": [[0.0, 0.0, 0.0]]}).observes()
    assert [spec.name for spec in requirement.channels] == ["state"]


def test_an_unusable_action_spec_is_refused_at_construction():
    with pytest.raises(ConfigError, match="action spec"):
        ServedPolicy(URL, ChannelSpec("action", "vector", (3, 3), "float32"))


# -- reading the response --------------------------------------------------


def test_a_chunk_comes_back_whole():
    chunk = policy({"action": [[1, 2, 3], [4, 5, 6]]}).act(observation())
    assert chunk.shape == (2, 3)
    assert chunk.dtype == np.dtype("float32")


def test_a_single_action_is_a_chunk_of_one():
    """Rank fixes this: the spec says an action is rank 1, so rank 1 is one action."""
    assert policy({"action": [1, 2, 3]}).act(observation()).shape == (1, 3)


def test_a_bare_array_response_needs_no_envelope():
    assert policy([[1, 2, 3]]).act(observation()).shape == (1, 3)


def test_a_custom_envelope_is_a_constructor_argument():
    served = ServedPolicy(
        URL,
        ACTION,
        [STATE],
        chunk=2,
        transport=transport_returning({"result": {"actions": [[1, 2, 3]]}}),
        decode=lambda payload: payload["result"]["actions"],
    )
    assert served.act(observation()).shape == (1, 3)


def test_the_observation_goes_out_under_its_own_name():
    calls: list = []
    served = policy({"action": [[0, 0, 0]]}, calls=calls)
    served.reset(EpisodeContext("scene-1", instruction="do the thing", seed=7))
    served.act(Observation(3, {"state": np.array([1.0, 2.0, 3.0], dtype="float32")}))
    sent = calls[-1]["body"]
    assert sent["observation"]["state"] == [1.0, 2.0, 3.0]
    assert (sent["step"], sent["episode_id"], sent["seed"]) == (3, "scene-1", 7)
    assert sent["instruction"] == "do the thing"


def test_headers_travel_with_every_call():
    calls: list = []
    served = ServedPolicy(
        Endpoint(URL, headers={"Authorization": "Bearer x"}),
        ACTION,
        [STATE],
        transport=transport_returning({"action": [[0, 0, 0]]}, calls=calls),
    )
    served.act(observation())
    assert calls[-1]["headers"]["Authorization"] == "Bearer x"


def test_reset_reaches_the_server_when_there_is_somewhere_to_send_it():
    calls: list = []
    served = ServedPolicy(
        Endpoint(URL, reset_url="http://policy.invalid/reset"),
        ACTION,
        [STATE],
        transport=transport_returning({"ok": True}, calls=calls),
    )
    served.reset(EpisodeContext("scene-1", seed=3))
    assert calls[-1]["url"].endswith("/reset")
    assert calls[-1]["body"]["seed"] == 3


def test_reset_stays_local_when_there_is_not():
    calls: list = []
    served = policy({"action": [[0, 0, 0]]}, calls=calls)
    served.reset(EpisodeContext("scene-1"))
    assert calls == []


# -- refusing a response that does not fit ---------------------------------


def test_a_chunk_longer_than_declared_is_refused_not_truncated():
    served = policy({"action": [[1, 2, 3]] * 6}, chunk=4)
    with pytest.raises(ComponentError, match="declares a chunk of 4"):
        served.act(observation())


def test_the_wrong_width_is_refused():
    with pytest.raises(ComponentError, match="action"):
        policy({"action": [[1, 2, 3, 4, 5]]}).act(observation())


def test_an_empty_response_is_refused():
    with pytest.raises(ComponentError, match="no actions"):
        policy({"action": []}).act(observation())


def test_too_many_axes_is_refused():
    with pytest.raises(ComponentError, match="rank"):
        policy({"action": [[[1, 2, 3]]]}).act(observation())


def test_a_missing_action_key_says_what_was_there():
    with pytest.raises(ComponentError, match="could not read"):
        policy({"prediction": [[1, 2, 3]]}).act(observation())


def test_a_response_that_is_not_json_is_one_lost_trial():
    def transport(url, body, headers, timeout):
        return b"<html>gateway</html>"

    served = ServedPolicy(URL, ACTION, [STATE], transport=transport)
    with pytest.raises(ComponentError, match="not JSON"):
        served.act(observation())


# -- the two meanings of failure -------------------------------------------


def test_a_server_that_never_answered_is_a_configuration_failure():
    """Otherwise a typo'd port produces a full run of uniform failures."""
    served = policy(urllib.error.URLError("connection refused"))
    with pytest.raises(ConfigError, match="has not answered once"):
        served.act(observation())


def test_a_server_that_stops_answering_costs_one_trial():
    served = policy({"action": [[1, 2, 3]]}, urllib.error.URLError("gone"))
    served.act(observation())
    with pytest.raises(ComponentError, match="did not answer"):
        served.act(observation())


def test_a_rejected_request_halts_because_it_will_be_rejected_again():
    served = policy(_http_error(422, "Unprocessable"))
    with pytest.raises(ConfigError, match="422"):
        served.act(observation())


def test_a_server_error_is_recoverable_because_the_server_answered():
    served = policy(_http_error(503, "Service Unavailable"))
    with pytest.raises(ComponentError):
        served.act(observation())


def _http_error(code: int, reason: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(URL, code, reason, {}, None)


def test_retries_are_declared_and_used():
    served = ServedPolicy(
        Endpoint(URL, attempts=3),
        ACTION,
        [STATE],
        chunk=2,
        transport=transport_returning(urllib.error.URLError("blip"), {"action": [[1, 2, 3]]}),
    )
    assert served.act(observation()).shape == (1, 3)


def test_attempts_below_one_is_refused():
    with pytest.raises(ValueError, match="attempts"):
        Endpoint(URL, attempts=0)


# -- the default transport, over a real socket -----------------------------


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - http.server's spelling
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length))
        state = request["observation"]["state"]
        body = json.dumps({"action": [[value + 1 for value in state]]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # noqa: D102 - quiet during tests
        pass


@pytest.fixture
def server():
    httpd = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}/act"
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=5)


def test_the_default_transport_speaks_to_a_real_server(server):
    served = ServedPolicy(server, ACTION, [STATE], chunk=1, deterministic=True)
    chunk = served.act(Observation(0, {"state": np.array([1.0, 2.0, 3.0], dtype="float32")}))
    assert np.allclose(chunk, [[2.0, 3.0, 4.0]])
    assert check_policy(served, [observation()]).ok


def test_a_dead_port_is_a_configuration_failure():
    """Port 1 is privileged and unbound here; nothing will ever answer on it."""
    served = ServedPolicy("http://127.0.0.1:1/act", ACTION, [STATE], chunk=1)
    with pytest.raises(ConfigError, match="has not answered once"):
        served.act(observation())


# -- constructible from a file ---------------------------------------------


def test_everything_that_has_to_cross_a_manifest_is_plain_data():
    """The config below is exactly what a manifest would carry, JSON and no more."""
    from gantry.serial import spec_to_dict

    config = {
        "endpoint": {"url": URL, "timeout": 5.0, "attempts": 2},
        "action": spec_to_dict(ACTION),
        "observes": [spec_to_dict(STATE)],
        "chunk": 2,
    }
    served = ServedPolicy(**config, transport=transport_returning({"action": [[1, 2, 3]]}))
    assert served.action_spec() == ACTION
    assert [spec.name for spec in served.observes().channels] == ["state"]
    assert served.endpoint.timeout == 5.0
    assert served.act(observation()).shape == (1, 3)


def test_a_load_bearing_key_survives_being_declared_as_json():
    from gantry.serial import spec_to_dict

    quaternion = ChannelSpec(
        "action",
        "vector",
        (4,),
        "float32",
        discriminators=("rotation_repr",),
        metadata={"rotation_repr": "quat_xyzw"},
    )
    served = ServedPolicy(
        URL, spec_to_dict(quaternion), transport=transport_returning({"action": [[0, 0, 0, 1]]})
    )
    assert served.action_spec().discriminators == ("rotation_repr",)


# -- the envelope helpers stand alone --------------------------------------


def test_encode_survives_a_missing_context():
    payload = encode_observation(observation(2), None)
    assert payload == {"step": 2, "observation": {"state": [0.0, 0.0, 0.0]}}


def test_decode_passes_a_bare_payload_through():
    assert decode_action([[1, 2, 3]]) == [[1, 2, 3]]
    assert decode_action({"action": 5}) == 5
    with pytest.raises(KeyError, match="action"):
        decode_action({"other": 1})
