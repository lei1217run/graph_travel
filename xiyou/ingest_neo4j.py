import os
import json
from typing import Dict
from neo4j import GraphDatabase
from xiyou.config import load_settings, resolve_paths, list_target_books
from xiyou.normalize_adapter import normalize_output, calibrate_relations, fuse_relations, collect_entities


class Neo4jIngestor:
    def __init__(self, settings: Dict):
        self.settings = settings
        self.driver = self._get_driver()

    def _get_driver(self):
        uri = self.settings.get("neo4j", {}).get("uri")
        user = self.settings.get("neo4j", {}).get("user")
        password = self.settings.get("neo4j", {}).get("password")
        return GraphDatabase.driver(uri, auth=(user, password))

    def ensure_constraints(self):
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT entity_unique IF NOT EXISTS FOR (e:Entity) REQUIRE (e.name, e.type) IS UNIQUE")
            session.run("CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
            session.run("CREATE CONSTRAINT event_unique IF NOT EXISTS FOR (e:Event) REQUIRE (e.event_type, e.book, e.chapter) IS UNIQUE")
            session.run("CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)")

    def neo4j_upsert(self, session, kg: Dict, book_key: str, chapter_id: str):
        for ent in kg.get("entities", []):
            session.run("MERGE (e:Entity {name:$name})", name=ent["name"]) 
        for rel in kg.get("relations", []):
            rt = rel.get("relation")
            ql_json = json.dumps(rel.get("qualifiers") or {}, ensure_ascii=False)
            session.run(
                "MATCH (h:Entity {name:$h}) MATCH (t:Entity {name:$t}) "
                "MERGE (h)-[r:RELATION {type:$rt, book:$bk, chapter:$cid}]->(t) "
                "SET r.confidence=$conf, r.evidence=$ev, r.qualifiers_json=$ql_json",
                h=rel["head"], t=rel["tail"],
                rt=rt, bk=book_key, cid=(rel.get("chapter_id") or chapter_id),
                conf=rel.get("confidence"), ev=rel.get("evidence"), ql_json=ql_json
            )
        for evt in kg.get("events", []):
            parts_json = json.dumps(evt.get("participants") or {}, ensure_ascii=False)
            session.run(
                "MERGE (e:Event {event_type:$et, book:$bk, chapter:$cid}) "
                "SET e.time=$tm, e.location=$loc, e.evidence=$ev, e.confidence=$conf, e.participants_json=$parts_json",
                et=evt.get("event_type"), bk=book_key, cid=chapter_id,
                tm=evt.get("time"), loc=evt.get("location"), ev=evt.get("evidence"),
                conf=evt.get("confidence"), parts_json=parts_json
            )

    def neo4j_upsert_events(self, session, events: list, book_key: str, chapter_id: str):
        for evt in events or []:
            parts_json = json.dumps(evt.get("participants") or {}, ensure_ascii=False)
            session.run(
                "MERGE (e:Event {event_type:$et, book:$bk, chapter:$cid}) "
                "SET e.time=$tm, e.location=$loc, e.evidence=$ev, e.confidence=$conf, e.participants_json=$parts_json",
                et=evt.get("event_type"), bk=book_key, cid=chapter_id,
                tm=evt.get("time"), loc=evt.get("location"), ev=evt.get("evidence"),
                conf=evt.get("confidence"), parts_json=parts_json
            )

    def ingest_results(self):
        books = list_target_books(self.settings)
        default_tpl = self.settings.get("prompts", {}).get("selected", "relations_plus")
        for bk in books:
            p = resolve_paths(self.settings, bk)
            base = p["results_dir"]
            names = [x for x in os.listdir(base) if x.endswith(".json")]
            all_rel = []
            events_list = []
            for name in names:
                fp = os.path.join(base, name)
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                meta = data.get("meta") or {}
                tpl_id = meta.get("template_id") or default_tpl
                tpl = next((t for t in self.settings.get("prompts", {}).get("templates", []) if t.get("id") == tpl_id), {})
                syn = tpl.get("variables", {}).get("synonyms_map", {}) or {}
                alias_rules = tpl.get("variables", {}).get("alias_rules", []) or []
                alias_relations = self.settings.get("relations", {}).get("alias_relations", [])
                kg = normalize_output(data, tpl_id, syn, alias_rules, alias_relations)
                chapter_id = meta.get("chapter_id") or name.split("result_")[1].split(".json")[0].split("__")[0]
                for r in kg.get("relations", []):
                    nr = dict(r)
                    nr["chapter_id"] = chapter_id
                    all_rel.append(nr)
                events_list.append((chapter_id, kg.get("events", [])))
            cal_rel = calibrate_relations(all_rel, self.settings)
            fused_rel = fuse_relations(cal_rel, self.settings)
            entities = []
            seen = set()
            for r in fused_rel:
                h = r.get("head")
                t = r.get("tail")
                if h and h not in seen:
                    seen.add(h); entities.append({"name": h})
                if t and t not in seen:
                    seen.add(t); entities.append({"name": t})
            book_key = bk
            with self.driver.session() as session:
                kg_rel = {"entities": entities, "relations": fused_rel, "events": []}
                self.neo4j_upsert(session, kg_rel, book_key, "MULTI")
                for chapter_id, evs in events_list:
                    self.neo4j_upsert_events(session, evs, book_key, chapter_id)

    def close(self):
        self.driver.close()


def ingest_results():
    settings = load_settings()
    ingestor = Neo4jIngestor(settings)
    ingestor.ensure_constraints()
    ingestor.ingest_results()
    ingestor.close()


def main():
    settings = load_settings()
    ingestor = Neo4jIngestor(settings)
    ingestor.ensure_constraints()
    ingestor.ingest_results()
    ingestor.close()


if __name__ == "__main__":
    main()
