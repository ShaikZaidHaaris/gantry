"""The protocol: the codec, the envelope, and what each failure means."""

from __future__ import annotations

import numpy as np
import pytest
import zmq
from gantry_policy_gr00t import Client, Codec, Endpoint
from server import FakeServer, modality_payload

from gantry.errors import ComponentError, ConfigError

MODALITY = {
    "state": modality_payload([0], ["x"]),
    "action": modality_payload([0, 1], ["x"]),
    "video": modality_payload([0], []),
    "language": modality_payload([0], ["annotation.human.task"]),
}


def echo(observation, options):
    return {"x": np.zeros((1, 2, 1), dtype="float32")}


@pytest.fixture
def server():
    fake = FakeServer(MODALITY, echo)
    yield fake
    fake.stop()


@pytest.fixture
def client(server):
    c = Client(Endpoint(port=server.port, timeout_ms=3000))
    yield c
    c.close()


# -- the codec -------------------------------------------------------------


def test_arrays_survive_the_round_trip():
    payload = {"state": {"x": np.arange(6, dtype="float32").reshape(2, 3)}}
    restored = Codec.from_bytes(Codec.to_bytes(payload))
    assert np.array_equal(restored["state"]["x"], payload["state"]["x"])
    assert restored["state"]["x"].dtype == np.float32


def test_a_modality_config_arrives_as_plain_data():
    """Rebuilding the dataclass would mean importing the package we avoid."""
    decoded = Codec.from_bytes(Codec.to_bytes(modality_payload([-1, 0], ["ego_view"])))
    assert decoded == {"delta_indices": [-1, 0], "modality_keys": ["ego_view"]}


def test_a_half_written_modality_payload_is_refused():
    with pytest.raises(ValueError, match="as_json"):
        Codec.from_bytes(Codec.to_bytes({"__ModalityConfig__": True}))


def test_an_object_array_is_refused_on_the_way_out():
    with pytest.raises(TypeError, match="object-dtype"):
        Codec.to_bytes({"x": np.array([{"a": 1}], dtype=object)})


def test_an_object_array_is_refused_on_the_way_in():
    """The bytes came off a socket, so unpickling them is not an option."""
    forged = Codec.to_bytes({"nd": 1, "kind": "O", "data": b"anything"})
    with pytest.raises(ValueError, match="object-dtype"):
        Codec.from_bytes(forged)


# -- the envelope ----------------------------------------------------------


def test_it_pings(client):
    assert client.ping() is True


def test_the_modality_config_comes_back_unwrapped(client):
    config = client.modality_config()
    assert config["action"] == {"delta_indices": [0, 1], "modality_keys": ["x"]}


def test_get_action_unpacks_the_action_and_info_pair(client):
    action, info = client.get_action({"state": {"x": np.zeros((1, 1, 1), dtype="float32")}})
    assert action["x"].shape == (1, 2, 1)
    assert info == {}


def test_the_request_carries_the_endpoint_and_the_data(client, server):
    client.get_action({"state": {"x": np.ones((1, 1, 1), dtype="float32")}})
    assert np.array_equal(server.seen[-1]["state"]["x"], np.ones((1, 1, 1)))


def test_reset_reaches_the_server(client, server):
    client.reset({"episode_id": "scene-1", "seed": 4})
    assert server.resets[-1] == {"episode_id": "scene-1", "seed": 4}


def test_an_api_token_travels_with_the_request(server):
    client = Client(Endpoint(port=server.port, timeout_ms=3000, api_token="secret"))
    assert client.ping() is True
    client.close()


# -- what a failure means --------------------------------------------------


def test_a_server_side_error_costs_one_trial(client):
    with pytest.raises(ComponentError, match="Unknown endpoint"):
        client.call("no_such_endpoint", {})


def test_nothing_listening_is_a_configuration_failure():
    """A typo'd port would otherwise look exactly like a model that fails every scene."""
    client = Client(Endpoint(port=1, timeout_ms=200))
    with pytest.raises(ConfigError, match="is the GR00T server running"):
        client.call("ping")
    client.close()


def test_a_server_that_answered_and_then_went_quiet_costs_one_trial():
    client = Client(Endpoint(port=1, timeout_ms=200))
    client.answered = True
    with pytest.raises(ComponentError, match="did not answer"):
        client.call("ping")
    client.close()


def test_a_timeout_leaves_the_socket_usable(server):
    """A REQ socket that timed out is waiting for a reply that will never come.

    Without rebuilding it, every later call fails for a reason that has nothing
    to do with the model, and one slow inference poisons the whole run.
    """
    client = Client(Endpoint(port=1, timeout_ms=150))
    with pytest.raises(ConfigError):
        client.call("ping")
    client.endpoint = Endpoint(port=server.port, timeout_ms=3000)
    client._connect()
    assert client.ping() is True
    client.close()


def test_ping_reports_false_rather_than_raising():
    client = Client(Endpoint(port=1, timeout_ms=150))
    assert client.ping() is False
    client.close()


def test_a_timeout_must_be_positive():
    with pytest.raises(ValueError, match="timeout_ms"):
        Endpoint(timeout_ms=0)


def test_the_address_is_the_one_gr00t_binds():
    assert Endpoint().address == "tcp://localhost:5555"
    assert zmq  # the transport is real, not mocked
