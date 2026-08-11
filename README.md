# Rig-libSQL

A [Rig](https://github.com/0xPlaygrounds/rig) vector store backed by [libSQL],
the open-contribution fork of SQLite maintained by Turso. It uses libSQL's
**built-in native vector support** (`FLOAT32(N)` columns, `libsql_vector_idx`
indexes, and the `vector_top_k` / `vector_distance_*` SQL functions), so unlike
`rig-sqlite` there is **no extension to load** — hand the store an async
`libsql::Connection` and it manages the rest.

[libSQL]: https://github.com/tursodatabase/libsql

> This is a standalone companion crate. It depends on the published
> [`rig-core`](https://crates.io/crates/rig-core) from crates.io — it does not
> require a local checkout of the rig repository.

## Usage

Add the crate to your `Cargo.toml`:

```toml
[dependencies]
rig-libsql = { path = "..." }   # or git = "..." once you publish it
rig-core = "0.41"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
libsql = "0.9"
```

See [`examples/vector_search_libsql.rs`](./examples/vector_search_libsql.rs)
for a complete runnable example.

## Opening a connection

The store takes any async [`libsql::Connection`] produced by a
[`libsql::Builder`]:

```rust
# async fn open() -> anyhow::Result<()> {
// Local file (or ":memory:")
let db = libsql::Builder::new_local("vector_store.db").build().await?;
let conn = db.connect()?;

// ...or a remote Turso database
let db = libsql::Builder::new_remote("libsql://<your-db>.turso.io", "<token>")
    .build()
    .await?;
let remote_conn = db.connect()?;
# Ok(())
# }
```

## Defining a document table

Implement `LibsqlVectorStoreTable` to declare the backing document table. The
store manages its own `<table>_embeddings` and `<table>_embedding_map` tables
for similarity search, so the schema only describes the user-facing document
columns (one of which must be named `id`).

```rust
use rig_core::Embed;
use rig_libsql::{Column, ColumnValue, LibsqlVectorStoreTable};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Embed, Serialize)]
struct Document {
    id: String,
    #[embed]
    text: String,
}

impl LibsqlVectorStoreTable for Document {
    fn name() -> &'static str {
        "documents"
    }

    fn schema() -> Vec<Column> {
        vec![
            Column::new("id", "TEXT PRIMARY KEY"),
            Column::new("text", "TEXT"),
        ]
    }

    fn id(&self) -> String {
        self.id.clone()
    }

    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
        vec![
            ("id", Box::new(self.id.clone())),
            ("text", Box::new(self.text.clone())),
        ]
    }
}
```

## Storing JSON Metadata

Declare JSON metadata columns with `Column::new("metadata", "JSON")` and store
the value as `serde_json::Value`. Rig writes the value as JSON text and parses
it back as structured JSON when documents are returned from vector searches.

## Filtering

libSQL's `vector_top_k` cannot apply arbitrary predicates during candidate
retrieval, so every filter expression is rendered as a document-table predicate
applied after candidate search. An exhaustive candidate limit is used so results
stay correct. Keys may reference plain columns (`category`) or SQLite JSON
expressions (`metadata->>'$.source'`):

```rust
use rig_core::vector_store::request::VectorSearchRequest;
use rig_libsql::LibsqlSearchFilter;

let req = VectorSearchRequest::builder()
    .query("release notes")
    .samples(5)
    .filter(LibsqlSearchFilter::eq(
        "metadata->>'$.source'",
        serde_json::json!("docs"),
    ))
    .build();
```

Use `->>` when you want SQLite to compare a JSON value as a SQL scalar (text,
number, boolean). Use `->` when you want to compare against JSON text.

## Distance metrics

`LibsqlDistanceMetric::Cosine` (default) and `LibsqlDistanceMetric::Euclidean`
select the scoring function used for candidate scoring, thresholding, and
ordering. Scores are always higher-is-better: cosine yields similarity
(`1 - cosine_distance`), euclidean yields the negative distance.
