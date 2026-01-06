import requests
import json
def _wb_search_entity(name):
        

        url = "https://www.wikidata.org/w/api.php"
        headers = {
                "User-Agent": "cekruger (cekruger99@gmail.com)"
        }
        params = {
            "action": "wbsearchentities",
            "search": name,
            "language": "en",
            "type": 'item',
            "format": "json",
            "limit": 5,
        }

        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            r.raise_for_status()
            hits = r.json().get("search", [])
            qid = hits[0]["id"] if hits else None
        except Exception:
            qid = None

        
        return qid


print(_wb_search_entity('Mount Lucania'))