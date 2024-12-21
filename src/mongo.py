from pymongo import MongoClient, errors
import threading
import logging
from rich.console import Console
from rich.logging import RichHandler

class MongoConnect(metaclass=SingletonMeta):
    """
    Singleton class to interact with MongoDB server.
    """
    def __init__(self, config_dict: dict):
        """
        Initializes the MongoConnect instance with the provided configuration.

        :param config_dict: Dictionary containing configuration parameters.
        """
        self.config = config_dict
        self.console = Console()
        self.logger = self.setup_logging(self.config.get("verbose", False))

        try:
            # Build the MongoDB URI
            uri = self.build_uri()
            self.client = MongoClient(
                uri,
                connectTimeoutMS=self.config.get("connectTimeoutMS", 30000),
                serverSelectionTimeoutMS=self.config.get("serverSelectionTimeoutMS", 30000)
            )
            # Trigger a server selection to verify connection
            self.client.admin.command('ping')
            self.logger.info("Successfully connected to MongoDB server.")
        except errors.ConfigurationError as ce:
            self.logger.error(f"Configuration error: {ce}")
            raise
        except errors.ConnectionFailure as cf:
            self.logger.error(f"Could not connect to MongoDB server: {cf}")
            raise
        except Exception as e:
            self.logger.error(f"An error occurred while connecting to MongoDB: {e}")
            raise

    def setup_logging(self, verbose: bool):
        """
        Sets up logging configuration with RichHandler.
        If verbose is True, set log level to DEBUG, else INFO.
        """
        log_level = logging.DEBUG if verbose else logging.INFO

        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=self.console)]
        )
        logger = logging.getLogger("MongoConnect")
        return logger

    def build_uri(self) -> str:
        """
        Constructs the MongoDB URI based on the configuration dictionary.

        :return: MongoDB URI string.
        """
        auth_part = ""
        if self.config.get("username") and self.config.get("password"):
            auth_part = f"{self.config['username']}:{self.config['password']}@"

        uri = f"mongodb://{auth_part}{self.config.get('host', 'localhost')}:{self.config.get('port', 27017)}"

        # Add authentication source and mechanism if provided
        params = []
        if self.config.get("authSource"):
            params.append(f"authSource={self.config['authSource']}")
        if self.config.get("authMechanism"):
            params.append(f"authMechanism={self.config['authMechanism']}")
        if self.config.get("ssl"):
            params.append(f"ssl=true")
            if self.config.get("ssl_ca_certs"):
                params.append(f"ssl_ca_certs={self.config['ssl_ca_certs']}")
        else:
            params.append(f"ssl=false")

        if params:
            uri += "?" + "&".join(params)

        return uri

    def list_databases(self) -> list:
        """
        Lists all databases in the MongoDB server.

        :return: List of database names.
        """
        self.logger.debug("Listing all databases.")
        try:
            databases = self.client.list_database_names()
            self.logger.info(f"Databases: {databases}")
            return databases
        except errors.PyMongoError as e:
            self.logger.error(f"Error listing databases: {e}")
            raise

    def create_database(self, db_name: str):
        """
        Creates a new database. Note that MongoDB creates a database implicitly when you create a collection or insert a document.

        :param db_name: Name of the database to create.
        """
        self.logger.debug(f"Creating database: {db_name}")
        try:
            # Create a dummy collection to create the database
            db = self.client[db_name]
            db.create_collection("dummy_collection")
            # Drop the dummy collection
            db.drop_collection("dummy_collection")
            self.logger.info(f"Database '{db_name}' created successfully.")
        except errors.CollectionInvalid:
            self.logger.warning(f"Database '{db_name}' already exists.")
        except errors.PyMongoError as e:
            self.logger.error(f"Error creating database '{db_name}': {e}")
            raise

    def list_collections(self, db_name: str) -> list:
        """
        Lists all collections in a specified database.

        :param db_name: Name of the database.
        :return: List of collection names.
        """
        self.logger.debug(f"Listing collections in database '{db_name}'.")
        try:
            db = self.client[db_name]
            collections = db.list_collection_names()
            self.logger.info(f"Collections in '{db_name}': {collections}")
            return collections
        except errors.PyMongoError as e:
            self.logger.error(f"Error listing collections in '{db_name}': {e}")
            raise

    def create_collection(self, db_name: str, collection_name: str):
        """
        Creates a new collection in a specified database.

        :param db_name: Name of the database.
        :param collection_name: Name of the collection to create.
        """
        self.logger.debug(f"Creating collection '{collection_name}' in database '{db_name}'.")
        try:
            db = self.client[db_name]
            db.create_collection(collection_name)
            self.logger.info(f"Collection '{collection_name}' created successfully in '{db_name}'.")
        except errors.CollectionInvalid:
            self.logger.warning(f"Collection '{collection_name}' already exists in '{db_name}'.")
        except errors.PyMongoError as e:
            self.logger.error(f"Error creating collection '{collection_name}' in '{db_name}': {e}")
            raise

    def insert_document(self, db_name: str, collection_name: str, document: dict):
        """
        Inserts a document into a specified collection.

        :param db_name: Name of the database.
        :param collection_name: Name of the collection.
        :param document: Document to insert.
        """
        self.logger.debug(f"Inserting document into '{collection_name}' in '{db_name}'.")
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            result = collection.insert_one(document)
            self.logger.info(f"Document inserted with ID: {result.inserted_id}")
            return result.inserted_id
        except errors.PyMongoError as e:
            self.logger.error(f"Error inserting document into '{collection_name}': {e}")
            raise

    def find_documents(self, db_name: str, collection_name: str, query: dict = {}) -> list:
        """
        Finds documents in a specified collection based on a query.

        :param db_name: Name of the database.
        :param collection_name: Name of the collection.
        :param query: Query dictionary to filter documents.
        :return: List of matching documents.
        """
        self.logger.debug(f"Finding documents in '{collection_name}' with query: {query}")
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            documents = list(collection.find(query))
            self.logger.info(f"Found {len(documents)} documents.")
            return documents
        except errors.PyMongoError as e:
            self.logger.error(f"Error finding documents in '{collection_name}': {e}")
            raise

    def update_documents(self, db_name: str, collection_name: str, query: dict, update: dict) -> int:
        """
        Updates documents in a specified collection based on a query.

        :param db_name: Name of the database.
        :param collection_name: Name of the collection.
        :param query: Query dictionary to filter documents.
        :param update: Update operations to apply.
        :return: Number of documents updated.
        """
        self.logger.debug(f"Updating documents in '{collection_name}' with query: {query} and update: {update}")
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            result = collection.update_many(query, update)
            self.logger.info(f"Number of documents updated: {result.modified_count}")
            return result.modified_count
        except errors.PyMongoError as e:
            self.logger.error(f"Error updating documents in '{collection_name}': {e}")
            raise

    def delete_documents(self, db_name: str, collection_name: str, query: dict) -> int:
        """
        Deletes documents in a specified collection based on a query.

        :param db_name: Name of the database.
        :param collection_name: Name of the collection.
        :param query: Query dictionary to filter documents.
        :return: Number of documents deleted.
        """
        self.logger.debug(f"Deleting documents in '{collection_name}' with query: {query}")
        try:
            db = self.client[db_name]
            collection = db[collection_name]
            result = collection.delete_many(query)
            self.logger.info(f"Number of documents deleted: {result.deleted_count}")
            return result.deleted_count
        except errors.PyMongoError as e:
            self.logger.error(f"Error deleting documents in '{collection_name}': {e}")
            raise

    def drop_collection(self, db_name: str, collection_name: str):
        """
        Drops a specified collection from a database.

        :param db_name: Name of the database.
        :param collection_name: Name of the collection to drop.
        """
        self.logger.debug(f"Dropping collection '{collection_name}' from database '{db_name}'.")
        try:
            db = self.client[db_name]
            db.drop_collection(collection_name)
            self.logger.info(f"Collection '{collection_name}' dropped successfully from '{db_name}'.")
        except errors.PyMongoError as e:
            self.logger.error(f"Error dropping collection '{collection_name}': {e}")
            raise

    def drop_database(self, db_name: str):
        """
        Drops a specified database.

        :param db_name: Name of the database to drop.
        """
        self.logger.debug(f"Dropping database '{db_name}'.")
        try:
            self.client.drop_database(db_name)
            self.logger.info(f"Database '{db_name}' dropped successfully.")
        except errors.PyMongoError as e:
            self.logger.error(f"Error dropping database '{db_name}': {e}")
            raise
