from atproto import Client
import os


def initialize_client():
    BLUESKY_IDENT = os.environ.get("BLUESKY_IDENT", "")
    BLUESKY_PASS = os.environ.get("BLUESKY_PASS", "")
    client = Client()
    client.login(BLUESKY_IDENT, BLUESKY_PASS)
    return client


def get_message(client, position=0):
    """returns last message from lemonde.fr using bluesky's api"""
    handle = "lemonde.fr"
    profile = client.app.bsky.actor.get_profile({"actor": handle})
    did = profile["did"]

    # Étape 3 : Récupérer les posts du compte (feed)
    feed = client.app.bsky.feed.get_author_feed({"actor": did, "limit": position + 1})

    # Étape 4 : Afficher le dernier post
    dernier_post = feed["feed"][position]["post"]
    dernier_message = dernier_post["record"]["text"]
    return dernier_message
