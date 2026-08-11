use rig_core::client::{EmbeddingsClient, ProviderClient};
use rig_core::providers::openai;
use rig_core::vector_store::request::VectorSearchRequest;
use rig_core::{
    Embed,
    embeddings::EmbeddingsBuilder,
    providers::openai::Client,
    vector_store::{InsertDocuments, VectorStoreIndex},
};
use rig_libsql::{
    Column, ColumnValue, LibsqlDistanceMetric, LibsqlVectorStore, LibsqlVectorStoreTable,
};
use serde::{Deserialize, Serialize};

#[derive(Embed, Clone, Debug, Deserialize, Serialize)]
struct Document {
    id: String,
    #[embed]
    content: String,
}

impl LibsqlVectorStoreTable for Document {
    fn name() -> &'static str {
        "documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("content", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("content", Box::new(self.content.clone())),
        ]
    }
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::DEBUG.into()),
        )
        .init();

    // Initialize OpenAI client
    let openai_client = Client::from_env()?;

    // Open an in-process libSQL database (swap `new_local` for `new_remote`
    // to target a Turso-hosted database with no other code changes).
    let db = libsql::Builder::new_local("vector_store.db")
        .build()
        .await?;
    let conn = db.connect()?;

    // Select the embedding model and generate our embeddings
    let model = openai_client.embedding_model(openai::TEXT_EMBEDDING_ADA_002);

    let documents = vec![
        Document {
            id: "doc0".to_string(),
            content: "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets".to_string(),
        },
        Document {
            id: "doc1".to_string(),
            content: "Definition of a *glarb-glarb*: A glarb-glarb is an ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.".to_string(),
        },
        Document {
            id: "doc2".to_string(),
            content: "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.".to_string(),
        },
    ];

    let embeddings = EmbeddingsBuilder::new(model.clone())
        .documents(documents)?
        .build()
        .await?;

    // Initialize the libSQL vector store
    let vector_store: LibsqlVectorStore<_, Document> =
        LibsqlVectorStore::with_distance_metric(conn, &model, LibsqlDistanceMetric::Cosine).await?;

    // Add embeddings to vector store
    vector_store.insert_documents(embeddings).await?;

    // Create a vector index on our vector store
    let index = vector_store.index(model);

    let query = "What is a linglingdong?";
    let samples = 1;
    let req = VectorSearchRequest::builder()
        .samples(samples)
        .query(query)
        .build();

    // Query the index
    let results = index
        .top_n::<Document>(req.clone())
        .await?
        .into_iter()
        .collect::<Vec<_>>();

    println!("Results: {results:?}");

    let id_results = index.top_n_ids(req).await?.into_iter().collect::<Vec<_>>();

    println!("ID results: {id_results:?}");

    Ok(())
}
