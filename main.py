from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine.base import Engine
from sqlalchemy import text
from collections import defaultdict
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
#Setup query engine
from llama_index.core.query_engine import RetrieverQueryEngine, SQLAutoVectorQueryEngine
from llama_index.core.tools import QueryEngineTool
from llama_index.core.retrievers import VectorIndexAutoRetriever
from llama_index.core.vector_stores import MetadataInfo, VectorStoreInfo

# model selection
model = "gpt-4"
chunk_size = 2048

# in-memory db, next: change this to mySql
engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
# engine = create_engine('mysql+pymydsql://sinanasa@localhost/mydb')

def add_df_to_sql_database(table_name: str, pandas_df: pd.DataFrame, engine: Engine) -> None:
    pandas_df.to_sql(table_name, engine)

# general configurations
llm = OpenAI(temperature=0, model=model, streaming=True)
# service_context = ServiceContext.from_defaults(llm=llm, chunk_size=chunk_size)

# configurations for vector store
# text_splitter = TokenTextSplitter(chunk_size=chunk_size)
# node_parser = SimpleNodeParser()
node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=20)

db = chromadb.PersistentClient(path="chroma_database")
chroma_collection = db.get_or_create_collection(
    "my_chroma_store"
)
vector_store = ChromaVectorStore(
    chroma_collection=chroma_collection
)
storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

if chroma_collection.count() > 0:
    # Rebuild the Index from the ChromaDB in future sessions
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context
    )
    print("DB found. Running using db to build index...")
else:
    print("No DB found. Running without db...")
    # allow the creation of an Index
    index = VectorStoreIndex([], storage_context=storage_context)

    # all_teams_df = pd.read_csv("/Users/sinanasa/llm/baseball_data/lahman/lahman_1871-2023_csv/TeamsFranchises.csv", encoding = "utf-8")
    all_teams_df = pd.read_csv("source_data/TeamsFranchises.csv", encoding="utf-8")
    all_players_df = pd.read_csv("source_data/People.csv", encoding = "ISO-8859-1")
    all_games_df = pd.read_csv("source_data/Appearances.csv", encoding = "utf-8")

    add_df_to_sql_database("all_team_data", all_teams_df, engine)
    add_df_to_sql_database("all_player_data", all_players_df, engine)
    add_df_to_sql_database("all_games_data", all_games_df, engine)

    # set your table names here
    table_names = ["all_team_data", "all_player_data", "all_games_data"]

    # verify fb
    # with engine.connect() as conn:
    #     result = conn.execute(text(
    #         "SELECT nameFirst, nameLast, bats, throws FROM all_player_data WHERE nameLast is 'Piazza'"))
    #     for row in result:
    #         print(row)

    team_map = defaultdict(str)

    # grab the teams in the `all_team_data` database
    with engine.connect() as conn:
        results = conn.execute(text("SELECT DISTINCT franchID, franchName FROM all_team_data where active='Y'"))
        team_name_list = [result.franchName for result in results]
        results = conn.execute(text("SELECT DISTINCT franchID, franchName FROM all_team_data where  active='Y'"))
        team_list = [result.franchID for result in results]
        for i in range(len(team_list)):
            team_map[team_list[i]] = team_name_list[i]
            # team_map[[result.franchID for result in results]] = [result.franchName for result in results]

    # print(team_name_list)
    # for team in team_list:
    #     print(team)

    team_name_list = [team_map[team].split("/")[0] for team in team_list]
    wiki_docs = WikipediaReader().load_data(pages=team_name_list, auto_suggest=False)

    # set the tables you wish to include here
    tables_to_include = []

    sql_database = SQLDatabase(engine, include_tables=table_names)
    sql_query_engine = NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=table_names
    )

    for team, wiki_doc in zip(team_list, wiki_docs):
        nodes = node_parser.get_nodes_from_documents([wiki_doc])
        for node in nodes:
            node.metadata = {"title" : team_map[team]}
        index.insert_nodes(nodes)


#Setup Query Engine
team_vector_store_info = VectorStoreInfo(
    content_info="articles about different MLB teams",
    metadata_info=[
        MetadataInfo(
            name="title",
            type="str",
            description="The name of the MLB team.")
    ]
)

team_vector_auto_retriever = VectorIndexAutoRetriever(
    index, vector_store_info=team_vector_store_info
)

team_retriever_query_engine = RetrieverQueryEngine.from_args(
    team_vector_auto_retriever, storage_context=storage_context
)

sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    description=(
        "Useful for translating a natural language query into an SQL query over tables containing:"
        "all_player_data, containing biographical information on MLB baseball players."
        "all_team_data, containing stats related to MLB baseball teams"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=team_retriever_query_engine,
    description=f"Useful for answering semantic questions about MLB Baseball Teams"
)

query_engine = SQLAutoVectorQueryEngine(
    sql_tool,
    vector_tool
)

# response = query_engine.query("Who is Mike Piazza?")
while True:
    user_message = input("You: ")
    if user_message.lower() == 'exit':
        print("Exiting chat...")
        break
    response = query_engine.query(user_message)
    print(f"Chatbot: {response}")