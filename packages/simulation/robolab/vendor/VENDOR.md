# Vendored OpenPI websocket glue

These are the minimal, dependency-light pieces of
[Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
needed to speak the RoboLab / Cosmos3 policy protocol (msgpack + NumPy over
WebSocket). They carry **no** JAX/torch dependency, so they drop cleanly into
both the robosuite sim image (client side) and the Cosmos3 policy-server image
(server side) without disturbing either image's pins.

- Upstream: `Physical-Intelligence/openpi`
- Pinned commit: `95aadc6b16d170e4b13ab2e0ac64fbf2d1bb8e31` (same pin as `packages/vla/openpi`)
- License: Apache-2.0 (see upstream `LICENSE`)

Files (verbatim from upstream, path noted):

| vendored path | upstream path |
| --- | --- |
| `openpi_client/msgpack_numpy.py` | `packages/openpi-client/src/openpi_client/msgpack_numpy.py` |
| `openpi_client/base_policy.py` | `packages/openpi-client/src/openpi_client/base_policy.py` |
| `openpi_client/websocket_client_policy.py` | `packages/openpi-client/src/openpi_client/websocket_client_policy.py` |
| `openpi_server/websocket_policy_server.py` | `src/openpi/serving/websocket_policy_server.py` |

Runtime deps (wheels, installed at container launch): `msgpack`, `websockets`,
`typing_extensions` (numpy already present in both images).

To use: put this `vendor/` dir on `PYTHONPATH`. The Cosmos3 RoboLab server
imports `openpi_server.websocket_policy_server.WebsocketPolicyServer`; our client
adapter imports `openpi_client.websocket_client_policy.WebsocketClientPolicy`.
