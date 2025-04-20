import os
from atproto import Client


def initialize_client():
    """Initialize and authenticate a Bluesky client using env credentials."""
    BLUESKY_IDENT = os.environ.get("BLUESKY_IDENT", "")
    BLUESKY_PASS = os.environ.get("BLUESKY_PASS", "")
    client = Client()
    client.login(BLUESKY_IDENT, BLUESKY_PASS)
    return client


def get_message(client, position=0):
    """Return a message from @lemonde.fr at a given feed position."""
    handle = "lemonde.fr"
    profile = client.app.bsky.actor.get_profile({"actor": handle})
    did = profile["did"]
    feed = client.app.bsky.feed.get_author_feed({"actor": did, "limit": position + 1})
    dernier_post = feed["feed"][position]["post"]
    return dernier_post["record"]["text"]
