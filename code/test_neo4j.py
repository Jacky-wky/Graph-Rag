import os
from dotenv import load_dotenv
from neo4j import GraphDatabase


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

loaded = load_dotenv(ENV_PATH, override=True)


NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


def main():
    print("ENV path:", ENV_PATH)
    print("ENV loaded:", loaded)
    print("ENV file exists:", os.path.exists(ENV_PATH))
    print("NEO4J_URI:", repr(NEO4J_URI))
    print("NEO4J_USER:", repr(NEO4J_USER))
    print("NEO4J_DATABASE:", repr(NEO4J_DATABASE))
    print("NEO4J_PASSWORD is None:", NEO4J_PASSWORD is None)
    print("NEO4J_PASSWORD length:", len(NEO4J_PASSWORD) if NEO4J_PASSWORD is not None else None)

    if not NEO4J_PASSWORD:
        raise ValueError(
            "NEO4J_PASSWORD is empty or not loaded. Please check your .env file."
        )

    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        with driver.session(database=NEO4J_DATABASE) as session:
            result = session.run("RETURN 'Neo4j connected' AS message")
            record = result.single()
            print(record["message"])

    finally:
        driver.close()


if __name__ == "__main__":
    main()