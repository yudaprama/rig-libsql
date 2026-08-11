//! libSQL vector store integration for Rig.
//!
//! This crate provides [`LibsqlVectorStore`] and [`LibsqlVectorIndex`] for
//! storing embedded documents in [libSQL] using its built-in native vector
//! support (`FLOAT32(N)` columns, `libsql_vector_idx` indexes, and the
//! `vector_top_k` / `vector_distance_*` SQL functions). Define document table
//! schemas by implementing [`LibsqlVectorStoreTable`].
//!
//! Unlike the `rig-sqlite` crate, no native extension needs to be loaded: libSQL
//! ships vector search built-in, so you only hand the store an asynchronous
//! [`libsql::Connection`] obtained from a [`libsql::Builder`].
//!
//! [libSQL]: https://github.com/tursodatabase/libsql

use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::ops::RangeInclusive;

use libsql::Connection;
use rig_core::Embed;
use rig_core::embeddings::{Embedding, EmbeddingModel};
use rig_core::vector_store::request::{FilterError, SearchFilter, VectorSearchRequest};
use rig_core::vector_store::{InsertDocuments, VectorStoreError, VectorStoreIndex};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use zerocopy::IntoBytes;

/// Wraps any backend error as a [`VectorStoreError::DatastoreError`].
fn datastore<E>(error: E) -> VectorStoreError
where
    E: std::error::Error + Send + Sync + 'static,
{
    VectorStoreError::DatastoreError(Box::new(error))
}

/// Serialize an embedding into the little-endian `f32` byte blob that libSQL's
/// `vector(?)` / `vector_distance_*` SQL functions expect.
fn embedding_to_le_bytes(embedding: &Embedding) -> Vec<u8> {
    embedding
        .vec
        .iter()
        .map(|x| *x as f32)
        .collect::<Vec<f32>>()
        .as_bytes()
        .to_vec()
}

/// Serialize a query vector into a little-endian `f32` byte blob.
fn query_to_le_bytes(query: &[f32]) -> Vec<u8> {
    query.as_bytes().to_vec()
}

/// Value that can be stored in a libSQL vector store document column.
///
/// Use [`serde_json::Value`] for columns declared as `JSON`.
pub trait ColumnValue: Send + Sync {
    /// Converts this value to a typed libSQL value.
    fn to_libsql_value(&self) -> libsql::Value;

    /// Returns the SQL type name for this value.
    fn column_type(&self) -> &'static str;
}

/// A document-table column declaration.
#[derive(Clone, Debug)]
pub struct Column {
    name: &'static str,
    col_type: &'static str,
}

impl Column {
    pub fn new(name: &'static str, col_type: &'static str) -> Self {
        Self { name, col_type }
    }

    pub fn name(&self) -> &'static str {
        self.name
    }

    pub fn col_type(&self) -> &'static str {
        self.col_type
    }
}

/// A document type that can be persisted in a [`LibsqlVectorStore`].
///
/// Implementations declare the backing document table (`name` + `schema`) and
/// how an instance maps to column values. The vector store itself owns the
/// companion `<table>_embeddings` and `<table>_embedding_map` tables used for
/// similarity search, so the schema only describes the user-facing document
/// columns (one of which must be named `id`).
///
/// ```rust
/// use rig_core::Embed;
/// use serde::{Deserialize, Serialize};
/// use rig_libsql::{Column, ColumnValue, LibsqlVectorStoreTable};
///
/// #[derive(Embed, Clone, Debug, Deserialize, Serialize)]
/// struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl LibsqlVectorStoreTable for Document {
///     fn name() -> &'static str {
///         "documents"
///     }
///
///     fn schema() -> Vec<Column> {
///         vec![
///             Column::new("id", "TEXT PRIMARY KEY"),
///             Column::new("content", "TEXT"),
///         ]
///     }
///
///     fn id(&self) -> String {
///         self.id.clone()
///     }
///
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
/// ```
pub trait LibsqlVectorStoreTable: Send + Sync + Clone {
    fn name() -> &'static str;
    fn schema() -> Vec<Column>;
    fn id(&self) -> String;
    fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)>;
}

/// Distance metric used by libSQL vector searches.
///
/// The metric is applied consistently to candidate scoring, thresholding, and
/// ordering. Returned scores are always higher-is-better: cosine yields
/// similarity (`1 - cosine_distance`), while euclidean yields the negative
/// distance.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub enum LibsqlDistanceMetric {
    /// Cosine similarity, returned as `1 - vector_distance_cos(a, b)`.
    #[default]
    Cosine,
    /// Negative euclidean (L2) distance: `-vector_distance_l2(a, b)`.
    Euclidean,
}

impl LibsqlDistanceMetric {
    fn distance_function(self) -> &'static str {
        match self {
            Self::Cosine => "vector_distance_cos",
            Self::Euclidean => "vector_distance_l2",
        }
    }

    fn score_expression(self, query_param: &str, embedding_expr: &str) -> String {
        let function = self.distance_function();
        match self {
            Self::Cosine => format!("(1 - {function}({query_param}, {embedding_expr}))"),
            Self::Euclidean => format!("(-{function}({query_param}, {embedding_expr}))"),
        }
    }
}

#[derive(Debug)]
struct LibsqlMissingIdColumn {
    table_name: String,
}

impl Display for LibsqlMissingIdColumn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "libSQL vector store table `{}` is missing an `id` column",
            self.table_name
        )
    }
}

impl std::error::Error for LibsqlMissingIdColumn {}

/// libSQL-backed vector store for a document table `T` using embedding model `E`.
///
/// Create with [`LibsqlVectorStore::new`] (cosine similarity) or
/// [`LibsqlVectorStore::with_distance_metric`], then call
/// [`LibsqlVectorStore::index`] to obtain a searchable [`LibsqlVectorIndex`].
#[derive(Clone)]
pub struct LibsqlVectorStore<E, T>
where
    E: EmbeddingModel + 'static,
    T: LibsqlVectorStoreTable + 'static,
{
    conn: Connection,
    distance_metric: LibsqlDistanceMetric,
    _phantom: PhantomData<(E, T)>,
}

impl<E, T> LibsqlVectorStore<E, T>
where
    E: EmbeddingModel + Clone + 'static,
    T: LibsqlVectorStoreTable + 'static,
{
    /// Creates a libSQL vector store using cosine similarity.
    pub async fn new(conn: Connection, embedding_model: &E) -> Result<Self, VectorStoreError> {
        Self::with_distance_metric(conn, embedding_model, LibsqlDistanceMetric::default()).await
    }

    /// Creates a libSQL vector store with the requested distance metric.
    ///
    /// The store manages its own schema: the user-declared document table plus a
    /// `<table>_embeddings` table (typed `FLOAT32(N)` vector column with a
    /// `libsql_vector_idx` index) and a `<table>_embedding_map` table linking
    /// each embedding row back to its document rowid. All three are created
    /// with `IF NOT EXISTS`, so the call is idempotent against an existing
    /// store.
    pub async fn with_distance_metric(
        conn: Connection,
        embedding_model: &E,
        distance_metric: LibsqlDistanceMetric,
    ) -> Result<Self, VectorStoreError> {
        let dims = embedding_model.ndims();
        let table_name = T::name();
        let embeddings_table_name = format!("{table_name}_embeddings");
        let embedding_map_table_name = format!("{table_name}_embedding_map");
        let embeddings_index_name = format!("{table_name}_embeddings_idx");

        let schema = T::schema();
        if !schema.iter().any(|column| column.name == "id") {
            return Err(datastore(LibsqlMissingIdColumn {
                table_name: table_name.to_string(),
            }));
        }

        let mut create_table = format!("CREATE TABLE IF NOT EXISTS {table_name} (");
        for (i, column) in schema.iter().enumerate() {
            if i > 0 {
                create_table.push(',');
            }
            create_table.push_str(&format!("\n    {} {}", column.name, column.col_type));
        }
        create_table.push_str("\n)");

        let schema_batch = format!(
            "{create_table};
             CREATE INDEX IF NOT EXISTS idx_{table_name}_id ON {table_name}(id);
             CREATE TABLE IF NOT EXISTS {embeddings_table_name} (
                 embedding FLOAT32({dims})
             );
             CREATE INDEX IF NOT EXISTS {embeddings_index_name}
                 ON {embeddings_table_name} (libsql_vector_idx(embedding));
             CREATE TABLE IF NOT EXISTS {embedding_map_table_name} (
                 embedding_rowid INTEGER PRIMARY KEY,
                 document_rowid INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_{table_name}_embedding_map_document_rowid
                 ON {embedding_map_table_name}(document_rowid);"
        );

        conn.execute_batch(&schema_batch).await.map_err(datastore)?;

        Ok(Self {
            conn,
            distance_metric,
            _phantom: PhantomData,
        })
    }

    pub fn index(self, model: E) -> LibsqlVectorIndex<E, T> {
        LibsqlVectorIndex::new(model, self)
    }

    /// Inserts documents with their precomputed embeddings, replacing any
    /// previous embeddings for documents that share the same `id`.
    pub async fn add_rows(
        &self,
        documents: Vec<(T, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        if documents.is_empty() {
            return Ok(());
        }

        info!("Adding {} documents to libSQL store", documents.len());

        let table_name = T::name();
        let embeddings_table_name = format!("{table_name}_embeddings");
        let embedding_map_table_name = format!("{table_name}_embedding_map");

        let tx = self.conn.transaction().await.map_err(datastore)?;

        for (doc, embeddings) in &documents {
            debug!("Storing document with id {}", doc.id());

            let values = doc.column_values();
            let id_value = values
                .iter()
                .find(|(name, _)| *name == "id")
                .map(|(_, value)| value.to_libsql_value())
                .unwrap_or_else(|| libsql::Value::Text(doc.id()));

            // Replace any previous embeddings for this document id.
            let existing_rowid: Option<i64> = {
                let mut rows = tx
                    .query(
                        &format!("SELECT rowid FROM {table_name} WHERE id = ?"),
                        vec![id_value.clone()],
                    )
                    .await
                    .map_err(datastore)?;
                if let Some(row) = rows.next().await.map_err(datastore)? {
                    row.get::<i64>(0).ok()
                } else {
                    None
                }
            };

            if let Some(document_rowid) = existing_rowid {
                tx.execute(
                    &format!(
                        "DELETE FROM {embeddings_table_name}
                         WHERE rowid IN (
                             SELECT embedding_rowid FROM {embedding_map_table_name}
                             WHERE document_rowid = ?
                         )"
                    ),
                    vec![libsql::Value::Integer(document_rowid)],
                )
                .await
                .map_err(datastore)?;
                tx.execute(
                    &format!("DELETE FROM {embedding_map_table_name} WHERE document_rowid = ?"),
                    vec![libsql::Value::Integer(document_rowid)],
                )
                .await
                .map_err(datastore)?;
            }

            let columns = values.iter().map(|(col, _)| *col).collect::<Vec<_>>();
            let placeholders = (1..=values.len())
                .map(|i| format!("?{i}"))
                .collect::<Vec<_>>()
                .join(", ");
            let insert_sql = format!(
                "INSERT OR REPLACE INTO {table_name} ({}) VALUES ({placeholders})",
                columns.join(", "),
            );

            let params = values
                .iter()
                .map(|(_, value)| value.to_libsql_value())
                .collect::<Vec<_>>();
            tx.execute(&insert_sql, params).await.map_err(datastore)?;
            let document_rowid = tx.last_insert_rowid();

            for embedding in embeddings {
                let blob = embedding_to_le_bytes(embedding);
                let mut rows = tx
                    .query(
                        &format!(
                            "INSERT INTO {embeddings_table_name} (embedding) VALUES (vector(?)) RETURNING rowid"
                        ),
                        vec![libsql::Value::Blob(blob)],
                    )
                    .await
                    .map_err(datastore)?;
                let embedding_rowid = match rows.next().await.map_err(datastore)? {
                    Some(row) => row.get::<i64>(0).map_err(datastore)?,
                    None => {
                        return Err(datastore(std::io::Error::other(
                            "libSQL embedding insert returned no rowid",
                        )));
                    }
                };
                tx.execute(
                    &format!(
                        "INSERT INTO {embedding_map_table_name} (embedding_rowid, document_rowid) VALUES (?, ?)"
                    ),
                    vec![
                        libsql::Value::Integer(embedding_rowid),
                        libsql::Value::Integer(document_rowid),
                    ],
                )
                .await
                .map_err(datastore)?;
            }
        }

        tx.commit().await.map_err(datastore)?;
        Ok(())
    }
}

impl<E, T> LibsqlVectorStore<E, T>
where
    E: EmbeddingModel + 'static,
    T: LibsqlVectorStoreTable + 'static,
{
    /// Returns the number of stored embeddings and distinct documents.
    async fn embedding_counts(&self) -> Result<(u64, u64), VectorStoreError> {
        let table_name = T::name();
        let embeddings_table = format!("{table_name}_embeddings");
        let map_table = format!("{table_name}_embedding_map");
        let mut rows = self
            .conn
            .query(
                &format!(
                    "SELECT
                        (SELECT COUNT(*) FROM {embeddings_table}),
                        (SELECT COUNT(DISTINCT document_rowid) FROM {map_table})"
                ),
                (),
            )
            .await
            .map_err(datastore)?;
        let row = rows.next().await.map_err(datastore)?.ok_or_else(|| {
            datastore(std::io::Error::other(
                "libSQL embedding-count query returned no rows",
            ))
        })?;
        let embeddings: i64 = row.get(0).map_err(datastore)?;
        let documents: i64 = row.get(1).map_err(datastore)?;
        Ok((
            u64::try_from(embeddings).unwrap_or(0),
            u64::try_from(documents).unwrap_or(0),
        ))
    }

    /// Computes how many candidates to retrieve from `vector_top_k` so that the
    /// requested `samples` survive post-filtering and per-document dedup.
    async fn candidate_limit(&self, samples: u64, has_post_filters: bool) -> u64 {
        if samples == 0 {
            return 0;
        }
        let (embedding_count, document_count) = self.embedding_counts().await.unwrap_or((0, 0));

        if has_post_filters {
            // Post-filters can discard any candidate; only an exhaustive
            // retrieval guarantees the requested number survives.
            embedding_count.max(samples)
        } else if embedding_count > document_count {
            // Some document owns multiple embeddings. After dedup-to-document
            // (keeping the best), guaranteeing the top-`samples` documents needs
            // `samples + extra-embeddings` candidates. This bound is tight and
            // never exceeds the total embedding count.
            samples
                .saturating_add(embedding_count - document_count)
                .min(embedding_count)
        } else {
            samples
        }
        .max(1)
    }

    fn distance_metric(&self) -> LibsqlDistanceMetric {
        self.distance_metric
    }
}

impl<E, T> InsertDocuments for LibsqlVectorStore<E, T>
where
    E: EmbeddingModel + Clone + Send + Sync + 'static,
    T: LibsqlVectorStoreTable + for<'de> Deserialize<'de> + Send + Sync + 'static,
{
    async fn insert_documents<Doc: Serialize + Embed + Send>(
        &self,
        documents: Vec<(Doc, Vec<Embedding>)>,
    ) -> Result<(), VectorStoreError> {
        if documents.is_empty() {
            return Ok(());
        }

        let rows = documents
            .into_iter()
            .map(|(document, embeddings)| {
                let document = serde_json::to_value(document)?;
                let row = serde_json::from_value::<T>(document)?;
                Ok::<(T, Vec<Embedding>), VectorStoreError>((row, embeddings))
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.add_rows(rows).await
    }
}

/// Search filter for libSQL vector searches.
///
/// Because libSQL's `vector_top_k` cannot apply arbitrary predicates during
/// candidate retrieval, every filter expression is rendered as a document-table
/// predicate and applied after candidate search (an exhaustive candidate limit
/// is used so results stay correct). Keys may reference plain columns
/// (`category`) or SQLite JSON expressions (`metadata->>'$.source'`).
#[derive(Clone, Deserialize, Serialize, Debug)]
pub struct LibsqlSearchFilter {
    expr: LibsqlSearchFilterExpr,
}

impl Default for LibsqlSearchFilter {
    fn default() -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Raw {
                condition: "1 = 1".to_string(),
                params: Vec::new(),
            },
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
enum LibsqlSearchFilterExpr {
    Comparison {
        key: String,
        op: LibsqlComparisonOp,
        value: serde_json::Value,
    },
    And(Box<LibsqlSearchFilterExpr>, Box<LibsqlSearchFilterExpr>),
    Or(Box<LibsqlSearchFilterExpr>, Box<LibsqlSearchFilterExpr>),
    Not(Box<LibsqlSearchFilterExpr>),
    Between {
        key: String,
        lo: serde_json::Value,
        hi: serde_json::Value,
    },
    NullCheck {
        key: String,
        negated: bool,
    },
    Pattern {
        key: String,
        op: LibsqlPatternOp,
        pattern: String,
    },
    Raw {
        condition: String,
        params: Vec<serde_json::Value>,
    },
}

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize, Debug)]
enum LibsqlComparisonOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
}

impl LibsqlComparisonOp {
    fn as_sql(self) -> &'static str {
        match self {
            Self::Eq => "=",
            Self::Ne => "!=",
            Self::Gt => ">",
            Self::Gte => ">=",
            Self::Lt => "<",
            Self::Lte => "<=",
        }
    }
}

#[derive(Clone, Copy, Deserialize, Serialize, Debug)]
enum LibsqlPatternOp {
    Glob,
    Like,
}

impl LibsqlPatternOp {
    fn as_sql(self) -> &'static str {
        match self {
            Self::Glob => "glob",
            Self::Like => "like",
        }
    }
}

impl SearchFilter for LibsqlSearchFilter {
    type Value = serde_json::Value;

    fn eq(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Comparison {
                key: key.as_ref().to_string(),
                op: LibsqlComparisonOp::Eq,
                value,
            },
        }
    }

    fn gt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Comparison {
                key: key.as_ref().to_string(),
                op: LibsqlComparisonOp::Gt,
                value,
            },
        }
    }

    fn lt(key: impl AsRef<str>, value: Self::Value) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Comparison {
                key: key.as_ref().to_string(),
                op: LibsqlComparisonOp::Lt,
                value,
            },
        }
    }

    fn and(self, rhs: Self) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::And(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }

    fn or(self, rhs: Self) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Or(Box::new(self.expr), Box::new(rhs.expr)),
        }
    }
}

impl LibsqlSearchFilter {
    #[allow(clippy::should_implement_trait)]
    /// Negates a filter.
    pub fn not(self) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Not(Box::new(self.expr)),
        }
    }

    /// Tests whether the value at `key` is contained in the inclusive range.
    pub fn between<N>(key: impl Into<String>, range: RangeInclusive<N>) -> Self
    where
        N: Into<serde_json::Value>,
    {
        let (lo, hi) = range.into_inner();
        Self {
            expr: LibsqlSearchFilterExpr::Between {
                key: key.into(),
                lo: lo.into(),
                hi: hi.into(),
            },
        }
    }

    pub fn is_null(key: impl Into<String>) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::NullCheck {
                key: key.into(),
                negated: false,
            },
        }
    }

    pub fn is_not_null(key: impl Into<String>) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::NullCheck {
                key: key.into(),
                negated: true,
            },
        }
    }

    /// Tests whether the value at `key` satisfies the `LIKE` pattern.
    pub fn like(key: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Pattern {
                key: key.into(),
                op: LibsqlPatternOp::Like,
                pattern: pattern.into(),
            },
        }
    }

    /// Tests whether the value at `key` satisfies the `GLOB` pattern.
    pub fn glob(key: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            expr: LibsqlSearchFilterExpr::Pattern {
                key: key.into(),
                op: LibsqlPatternOp::Glob,
                pattern: pattern.into(),
            },
        }
    }

    fn render(&self) -> Result<(String, Vec<libsql::Value>), FilterError> {
        self.expr.render()
    }
}

impl LibsqlSearchFilterExpr {
    fn render(&self) -> Result<(String, Vec<libsql::Value>), FilterError> {
        match self {
            Self::Comparison { key, op, value } => {
                let key = qualify_document_key(key)?;
                Ok((
                    format!("{} {} ?", key, op.as_sql()),
                    vec![filter_param(value.clone())?],
                ))
            }
            Self::And(lhs, rhs) => {
                let (l_cond, mut l_params) = lhs.render()?;
                let (r_cond, r_params) = rhs.render()?;
                l_params.extend(r_params);
                Ok((format!("({l_cond}) AND ({r_cond})"), l_params))
            }
            Self::Or(lhs, rhs) => {
                let (l_cond, mut l_params) = lhs.render()?;
                let (r_cond, r_params) = rhs.render()?;
                l_params.extend(r_params);
                Ok((format!("({l_cond}) OR ({r_cond})"), l_params))
            }
            Self::Not(expr) => {
                let (cond, params) = expr.render()?;
                Ok((format!("NOT ({cond})"), params))
            }
            Self::Between { key, lo, hi } => {
                let key = qualify_document_key(key)?;
                Ok((
                    format!("{key} BETWEEN ? AND ?"),
                    vec![filter_param(lo.clone())?, filter_param(hi.clone())?],
                ))
            }
            Self::NullCheck { key, negated } => {
                let key = qualify_document_key(key)?;
                let op = if *negated { "IS NOT NULL" } else { "IS NULL" };
                Ok((format!("{key} {op}"), Vec::new()))
            }
            Self::Pattern { key, op, pattern } => {
                let key = qualify_document_key(key)?;
                Ok((
                    format!("{key} {} ?", op.as_sql()),
                    vec![libsql::Value::Text(pattern.clone())],
                ))
            }
            Self::Raw { condition, params } => {
                let mapped = params
                    .iter()
                    .cloned()
                    .map(filter_param)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((condition.clone(), mapped))
            }
        }
    }
}

/// Qualify a key with the `d.` document-table alias unless the caller already
/// qualified it or supplied a JSON/function expression.
fn qualify_document_key(key: &str) -> Result<String, FilterError> {
    if key.contains('.') || key.contains('(') || key.contains(' ') || key.contains('?') {
        return Ok(key.to_string());
    }
    if !is_plain_identifier(key) {
        return Err(FilterError::TypeError(format!(
            "`{key}` is not a supported libSQL document filter key"
        )));
    }
    Ok(format!("d.{key}"))
}

fn is_plain_identifier(key: &str) -> bool {
    let mut chars = key.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|c| c == '_' || c.is_ascii_alphanumeric())
}

fn filter_param(value: serde_json::Value) -> Result<libsql::Value, FilterError> {
    use serde_json::Value::*;

    match value {
        Null => Ok(libsql::Value::Null),
        Bool(b) => Ok(libsql::Value::Integer(b as i64)),
        String(s) => Ok(libsql::Value::Text(s)),
        Number(n) => Ok(if let Some(value) = n.as_i64() {
            libsql::Value::Integer(value)
        } else if let Some(value) = n.as_u64() {
            let value = i64::try_from(value).map_err(|_| {
                FilterError::TypeError(format!(
                    "libSQL integer filter value `{n}` exceeds i64::MAX"
                ))
            })?;
            libsql::Value::Integer(value)
        } else if let Some(float) = n.as_f64() {
            libsql::Value::Real(float)
        } else {
            libsql::Value::Text(n.to_string())
        }),
        Array(arr) => {
            let blob =
                serde_json::to_vec(&arr).map_err(|e| FilterError::Serialization(e.to_string()))?;
            Ok(libsql::Value::Blob(blob))
        }
        Object(obj) => {
            let blob =
                serde_json::to_vec(&obj).map_err(|e| FilterError::Serialization(e.to_string()))?;
            Ok(libsql::Value::Blob(blob))
        }
    }
}

/// Materialized search predicates: the score threshold plus any document filters.
///
/// Both are applied in the outer query, where the precomputed score column
/// (`score_column`, e.g. `ranked.__rig_score`) and the document table (`d.*`)
/// are both in scope. The score itself is computed once inside the `scored_raw`
/// CTE, so the threshold references that column rather than re-evaluating the
/// distance function.
struct LibsqlRenderedFilter {
    has_filters: bool,
    document_clause: String,
    params: Vec<libsql::Value>,
}

/// Render the threshold (against the precomputed score column) plus the
/// request's document-table filter.
fn render_search_filters(
    req: &VectorSearchRequest<LibsqlSearchFilter>,
    score_column: &str,
) -> Result<LibsqlRenderedFilter, FilterError> {
    let mut conditions = Vec::new();
    let mut params = Vec::new();
    let mut has_filters = false;

    if let Some(threshold) = req.threshold() {
        conditions.push(format!("{score_column} >= ?"));
        params.push(libsql::Value::Real(threshold));
        has_filters = true;
    }

    if let Some(filter) = req.filter() {
        let (clause, mut filter_params) = filter.render()?;
        conditions.push(clause);
        params.append(&mut filter_params);
        has_filters = true;
    }

    let document_clause = if conditions.is_empty() {
        String::new()
    } else {
        format!(" AND {}", conditions.join(" AND "))
    };

    Ok(LibsqlRenderedFilter {
        has_filters,
        document_clause,
        params,
    })
}

/// libSQL vector store index for Rig.
///
/// Built from a [`LibsqlVectorStore`] via [`LibsqlVectorStore::index`]. The
/// index embeds the query text with the same embedding model used for inserts
/// and ranks stored documents by similarity.
///
/// # Example
/// ```no_run
/// use rig_core::{
///     client::EmbeddingsClient,
///     embeddings::EmbeddingsBuilder,
///     providers::openai::{Client, TEXT_EMBEDDING_ADA_002},
///     vector_store::{InsertDocuments, VectorStoreIndex},
///     Embed,
/// };
/// use rig_core::vector_store::request::VectorSearchRequest;
/// use rig_libsql::{
///     Column, ColumnValue, LibsqlDistanceMetric, LibsqlVectorStore, LibsqlVectorStoreTable,
/// };
/// use serde::{Deserialize, Serialize};
///
/// # async fn example() -> anyhow::Result<()> {
/// #[derive(Embed, Clone, Debug, Deserialize, Serialize)]
/// struct Document {
///     id: String,
///     #[embed]
///     content: String,
/// }
///
/// impl LibsqlVectorStoreTable for Document {
///     fn name() -> &'static str {
///         "documents"
///     }
///
///     fn schema() -> Vec<Column> {
///         vec![
///             Column::new("id", "TEXT PRIMARY KEY"),
///             Column::new("content", "TEXT"),
///         ]
///     }
///
///     fn id(&self) -> String {
///         self.id.clone()
///     }
///
///     fn column_values(&self) -> Vec<(&'static str, Box<dyn ColumnValue>)> {
///         vec![
///             ("id", Box::new(self.id.clone())),
///             ("content", Box::new(self.content.clone())),
///         ]
///     }
/// }
///
/// let db = libsql::Builder::new_local(":memory:").build().await?;
/// let conn = db.connect()?;
/// let openai_client = Client::new("YOUR_API_KEY")?;
/// let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
///
/// let vector_store: LibsqlVectorStore<_, Document> = LibsqlVectorStore::with_distance_metric(
///     conn,
///     &model,
///     LibsqlDistanceMetric::Cosine,
/// )
/// .await?;
///
/// let documents = vec![
///     Document { id: "doc1".to_string(), content: "Example document 1".to_string() },
///     Document { id: "doc2".to_string(), content: "Example document 2".to_string() },
/// ];
///
/// let embeddings = EmbeddingsBuilder::new(model.clone())
///     .documents(documents)?
///     .build()
///     .await?;
///
/// vector_store.insert_documents(embeddings).await?;
///
/// let index = vector_store.index(model);
/// let req = VectorSearchRequest::builder()
///     .query("Example query")
///     .samples(2)
///     .build();
/// let results = index.top_n::<Document>(req).await?;
/// # let _ = results;
/// # Ok(())
/// # }
/// # let _ = example();
/// ```
pub struct LibsqlVectorIndex<E, T>
where
    E: EmbeddingModel + 'static,
    T: LibsqlVectorStoreTable + 'static,
{
    store: LibsqlVectorStore<E, T>,
    embedding_model: E,
}

impl<E, T> LibsqlVectorIndex<E, T>
where
    E: EmbeddingModel + 'static,
    T: LibsqlVectorStoreTable,
{
    pub fn new(embedding_model: E, store: LibsqlVectorStore<E, T>) -> Self {
        Self {
            store,
            embedding_model,
        }
    }
}

impl<E, T> VectorStoreIndex for LibsqlVectorIndex<E, T>
where
    E: EmbeddingModel + Send + Sync + 'static,
    T: LibsqlVectorStoreTable + 'static,
{
    type Filter = LibsqlSearchFilter;

    async fn top_n<D>(
        &self,
        req: VectorSearchRequest<LibsqlSearchFilter>,
    ) -> Result<Vec<(f64, String, D)>, VectorStoreError>
    where
        D: for<'de> Deserialize<'de> + Send,
    {
        tracing::debug!("Finding top {} matches for query", req.samples());
        if req.samples() == 0 {
            return Ok(Vec::new());
        }

        let columns = T::schema();
        let id_column_index = columns
            .iter()
            .position(|column| column.name == "id")
            .ok_or_else(|| {
                datastore(LibsqlMissingIdColumn {
                    table_name: T::name().to_string(),
                })
            })?;

        let (sql, params) =
            build_top_n_query(&self.store, &self.embedding_model, &req, &columns, true).await?;

        let mut rows = self
            .store
            .conn
            .query(&sql, params)
            .await
            .map_err(datastore)?;

        let score_index = i32::try_from(columns.len()).unwrap_or(0);
        let id_column = columns.get(id_column_index).ok_or_else(|| {
            datastore(LibsqlMissingIdColumn {
                table_name: T::name().to_string(),
            })
        })?;

        let mut top_n = Vec::new();
        while let Some(row) = rows.next().await.map_err(datastore)? {
            let score: f64 = row.get::<f64>(score_index).map_err(datastore)?;
            let mut map = serde_json::Map::new();
            for (i, column) in columns.iter().enumerate() {
                let value = row.get::<libsql::Value>(i as i32).map_err(datastore)?;
                let value = column_value_to_json(i, column, value)?;
                map.insert(column.name.to_string(), value);
            }
            let id = id_to_string(id_column_index as i32, id_column, &row)?;
            let doc_value = serde_json::Value::Object(map);
            match serde_json::from_value::<D>(doc_value) {
                Ok(doc) => top_n.push((score, id, doc)),
                Err(e) => {
                    debug!("Failed to deserialize document {id}: {e}");
                    continue;
                }
            }
        }

        debug!("Returning {} matches", top_n.len());
        Ok(top_n)
    }

    async fn top_n_ids(
        &self,
        req: VectorSearchRequest<LibsqlSearchFilter>,
    ) -> Result<Vec<(f64, String)>, VectorStoreError> {
        tracing::debug!("Finding top {} document IDs for query", req.samples());
        if req.samples() == 0 {
            return Ok(Vec::new());
        }

        let columns = T::schema();
        let (sql, params) =
            build_top_n_query(&self.store, &self.embedding_model, &req, &columns, false).await?;

        let mut rows = self
            .store
            .conn
            .query(&sql, params)
            .await
            .map_err(datastore)?;

        let id_column_index = columns
            .iter()
            .position(|column| column.name == "id")
            .ok_or_else(|| {
                datastore(LibsqlMissingIdColumn {
                    table_name: T::name().to_string(),
                })
            })?;

        let mut results = Vec::new();
        // `top_n_ids` projects exactly `d.id` (column 0) then the score
        // (column 1), regardless of how many columns the document schema has.
        let id_column = columns.get(id_column_index).ok_or_else(|| {
            datastore(LibsqlMissingIdColumn {
                table_name: T::name().to_string(),
            })
        })?;
        while let Some(row) = rows.next().await.map_err(datastore)? {
            let id = id_to_string(0, id_column, &row)?;
            let score: f64 = row.get::<f64>(1).map_err(datastore)?;
            results.push((score, id));
        }

        debug!("Found {} matching document IDs", results.len());
        Ok(results)
    }
}

/// Build the `vector_top_k`-based search query shared by `top_n` / `top_n_ids`.
///
/// `include_document` selects whether the document columns are projected (for
/// `top_n`) or only `id` + score (for `top_n_ids`).
async fn build_top_n_query<E, T>(
    store: &LibsqlVectorStore<E, T>,
    embedding_model: &E,
    req: &VectorSearchRequest<LibsqlSearchFilter>,
    columns: &[Column],
    include_document: bool,
) -> Result<(String, Vec<libsql::Value>), VectorStoreError>
where
    E: EmbeddingModel + 'static,
    T: LibsqlVectorStoreTable + 'static,
{
    let embedding = embedding_model.embed_text(req.query()).await?;
    let query_vec: Vec<f32> = embedding.vec.iter().map(|x| *x as f32).collect();
    let query_blob = query_to_le_bytes(&query_vec);

    let table_name = T::name();
    let embeddings_table_name = format!("{table_name}_embeddings");
    let embedding_map_table_name = format!("{table_name}_embedding_map");
    let embeddings_index_name = format!("{table_name}_embeddings_idx");

    let distance_metric = store.distance_metric();
    // Score expression references the query vector as an anonymous `?` and the
    // candidate embedding as `e.embedding` (the embeddings table is the real
    // FROM target of `scored_raw`, so the column always resolves). It is
    // evaluated exactly once per candidate.
    let score_expression = distance_metric.score_expression("?", "e.embedding");
    let filter = render_search_filters(req, "ranked.__rig_score")?;

    let candidate_limit = store
        .candidate_limit(req.samples(), filter.has_filters)
        .await;

    let outer_select = if include_document {
        columns
            .iter()
            .map(|column| format!("d.{} AS {}", column.name, column.name))
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        "d.id".to_string()
    };

    // Candidate rowids come from `vector_top_k` (ANN over the
    // `libsql_vector_idx`); we then re-score every candidate exactly with the
    // scalar distance function so thresholding and ordering are precise even
    // though candidate retrieval is approximate. The score is computed once in
    // `scored_raw` and the per-document dedup (keeping the best embedding per
    // document) happens in `ranked`.
    let sql = format!(
        "WITH scored_raw AS (
            SELECT m.document_rowid AS __rig_document_rowid,
                   {score_expression} AS __rig_score
            FROM {embeddings_table_name} e
            JOIN {embedding_map_table_name} m ON e.rowid = m.embedding_rowid
            WHERE e.rowid IN (
                SELECT rowid FROM vector_top_k('{embeddings_index_name}', vector(?), ?)
            )
         ),
         ranked AS (
            SELECT __rig_document_rowid AS __rig_document_rowid,
                   __rig_score AS __rig_score,
                   ROW_NUMBER() OVER (
                       PARTITION BY __rig_document_rowid
                       ORDER BY __rig_score DESC, __rig_document_rowid ASC
                   ) AS __rig_rank
            FROM scored_raw
         )
         SELECT {outer_select}, ranked.__rig_score
         FROM ranked
         JOIN {table_name} d ON ranked.__rig_document_rowid = d.rowid
         WHERE ranked.__rig_rank = 1{document_clause}
         ORDER BY ranked.__rig_score DESC, d.id ASC
         LIMIT ?",
        document_clause = filter.document_clause
    );

    // Param order: [query_vec (score expr), query_vec (vector_top_k), k,
    //               ...filter params, limit]. The query vector is bound twice
    //   because libSQL positional `?` placeholders are not reused the way
    //   numbered `?1`/`?2` placeholders are, and we keep the dialect
    //   numbering-agnostic so Vec-based binding stays portable.
    let mut params = Vec::with_capacity(4 + filter.params.len());
    params.push(libsql::Value::Blob(query_blob.clone()));
    params.push(libsql::Value::Blob(query_blob));
    params.push(libsql::Value::Integer(
        i64::try_from(candidate_limit).unwrap_or(i64::MAX),
    ));
    params.extend(filter.params);
    params.push(libsql::Value::Integer(
        i64::try_from(req.samples()).unwrap_or(i64::MAX),
    ));

    Ok((sql, params))
}

/// Decode a stored libSQL value into JSON according to the declared column type.
fn column_value_to_json(
    index: usize,
    column: &Column,
    value: libsql::Value,
) -> Result<serde_json::Value, VectorStoreError> {
    use libsql::Value;

    if column_declares_json(column.col_type()) {
        return match value {
            Value::Null => Ok(serde_json::Value::Null),
            Value::Text(text) => decode_json_text(index, column, &text),
            Value::Blob(bytes) => {
                let text = std::str::from_utf8(&bytes).map_err(|e| {
                    datastore(LibsqlColumnValueError {
                        column_name: column.name,
                        column_type: column.col_type,
                        message: format!("invalid UTF-8 JSON text: {e}"),
                        index,
                    })
                })?;
                decode_json_text(index, column, text)
            }
            Value::Integer(value) => Ok(serde_json::Value::Number(value.into())),
            Value::Real(value) => number_value(index, column, value),
        };
    }

    let affinity = ColumnAffinity::from_column_type(column.col_type());
    match (affinity, value) {
        (_, Value::Null) => Ok(serde_json::Value::Null),
        (ColumnAffinity::Boolean, Value::Integer(0)) => Ok(serde_json::Value::Bool(false)),
        (ColumnAffinity::Boolean, Value::Integer(1)) => Ok(serde_json::Value::Bool(true)),
        (ColumnAffinity::Boolean, _) => Err(datastore(LibsqlColumnValueError {
            column_name: column.name,
            column_type: column.col_type,
            message: "stored libSQL boolean value must be 0 or 1".to_string(),
            index,
        })),
        (_, Value::Text(text)) => Ok(serde_json::Value::String(text)),
        (_, Value::Integer(value)) => Ok(serde_json::Value::Number(value.into())),
        (_, Value::Real(value)) => number_value(index, column, value),
        (_, Value::Blob(bytes)) => Ok(serde_json::to_value(bytes)?),
    }
}

fn number_value(
    index: usize,
    column: &Column,
    value: f64,
) -> Result<serde_json::Value, VectorStoreError> {
    serde_json::Number::from_f64(value)
        .map(serde_json::Value::Number)
        .ok_or_else(|| {
            datastore(LibsqlColumnValueError {
                column_name: column.name,
                column_type: column.col_type,
                message: "non-finite float value".to_string(),
                index,
            })
        })
}

fn decode_json_text(
    index: usize,
    column: &Column,
    text: &str,
) -> Result<serde_json::Value, VectorStoreError> {
    serde_json::from_str(text).map_err(|e| {
        datastore(LibsqlColumnValueError {
            column_name: column.name,
            column_type: column.col_type,
            message: format!("invalid JSON text: {e}"),
            index,
        })
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ColumnAffinity {
    Text,
    Integer,
    Float,
    Boolean,
    Numeric,
    Blob,
}

impl ColumnAffinity {
    fn from_column_type(column_type: &str) -> Self {
        let column_type = column_type.to_ascii_uppercase();
        if column_type.contains("INT") {
            Self::Integer
        } else if column_type.contains("CHAR")
            || column_type.contains("CLOB")
            || column_type.contains("TEXT")
        {
            Self::Text
        } else if column_type.contains("BLOB") || column_type.trim().is_empty() {
            Self::Blob
        } else if column_type.contains("REAL")
            || column_type.contains("FLOA")
            || column_type.contains("DOUB")
        {
            Self::Float
        } else if column_type.contains("BOOL") {
            Self::Boolean
        } else {
            Self::Numeric
        }
    }
}

fn column_declares_json(column_type: &str) -> bool {
    column_type
        .split_whitespace()
        .next()
        .is_some_and(|token| token.eq_ignore_ascii_case("JSON"))
}

#[derive(Debug)]
struct LibsqlColumnValueError {
    column_name: &'static str,
    column_type: &'static str,
    message: String,
    index: usize,
}

impl Display for LibsqlColumnValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "could not convert libSQL column `{}` (declared `{}`) at index {}: {}",
            self.column_name, self.column_type, self.index, self.message
        )
    }
}

impl std::error::Error for LibsqlColumnValueError {}

/// Read the document id from a result row.
///
/// `row_index` is the position of the id column in the result set (0 for
/// `top_n_ids`, which projects only `d.id` + score; the schema position of `id`
/// for `top_n`, which projects every document column). `id_column` is the
/// schema declaration of the id column, used for error reporting.
fn id_to_string(
    row_index: i32,
    id_column: &Column,
    row: &libsql::Row,
) -> Result<String, VectorStoreError> {
    let value = row.get::<libsql::Value>(row_index).map_err(datastore)?;
    match value {
        libsql::Value::Integer(value) => Ok(value.to_string()),
        libsql::Value::Real(value) => Ok(value.to_string()),
        libsql::Value::Text(value) => Ok(value),
        libsql::Value::Null | libsql::Value::Blob(_) => Err(datastore(LibsqlColumnValueError {
            column_name: id_column.name,
            column_type: "TEXT or INTEGER",
            message: "id cannot be NULL or BLOB".to_string(),
            index: usize::try_from(row_index).unwrap_or(0),
        })),
    }
}

impl ColumnValue for String {
    fn to_libsql_value(&self) -> libsql::Value {
        libsql::Value::Text(self.clone())
    }

    fn column_type(&self) -> &'static str {
        "TEXT"
    }
}

impl ColumnValue for i64 {
    fn to_libsql_value(&self) -> libsql::Value {
        libsql::Value::Integer(*self)
    }

    fn column_type(&self) -> &'static str {
        "INTEGER"
    }
}

impl ColumnValue for i32 {
    fn to_libsql_value(&self) -> libsql::Value {
        libsql::Value::Integer(i64::from(*self))
    }

    fn column_type(&self) -> &'static str {
        "INTEGER"
    }
}

impl ColumnValue for f64 {
    fn to_libsql_value(&self) -> libsql::Value {
        libsql::Value::Real(*self)
    }

    fn column_type(&self) -> &'static str {
        "FLOAT"
    }
}

impl ColumnValue for f32 {
    fn to_libsql_value(&self) -> libsql::Value {
        libsql::Value::Real(f64::from(*self))
    }

    fn column_type(&self) -> &'static str {
        "FLOAT"
    }
}

impl ColumnValue for bool {
    fn to_libsql_value(&self) -> libsql::Value {
        libsql::Value::Integer(if *self { 1 } else { 0 })
    }

    fn column_type(&self) -> &'static str {
        "BOOLEAN"
    }
}

impl ColumnValue for serde_json::Value {
    fn to_libsql_value(&self) -> libsql::Value {
        libsql::Value::Text(self.to_string())
    }

    fn column_type(&self) -> &'static str {
        "JSON"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_score_expression_uses_cosine_distance() {
        let expr = LibsqlDistanceMetric::Cosine.score_expression("?1", "e.embedding");
        assert_eq!(expr, "(1 - vector_distance_cos(?1, e.embedding))");
    }

    #[test]
    fn euclidean_score_expression_uses_l2_distance() {
        let expr = LibsqlDistanceMetric::Euclidean.score_expression("?1", "e.embedding");
        assert_eq!(expr, "(-vector_distance_l2(?1, e.embedding))");
    }

    #[test]
    fn eq_filter_qualifies_plain_key_with_document_alias() -> anyhow::Result<()> {
        let filter = LibsqlSearchFilter::eq("category", serde_json::json!("docs"));
        let (condition, params) = filter.render()?;
        anyhow::ensure!(
            condition == "d.category = ?",
            "unexpected condition: {condition}"
        );
        anyhow::ensure!(
            params == vec![libsql::Value::Text("docs".to_string())],
            "unexpected params: {params:?}"
        );
        Ok(())
    }

    #[test]
    fn json_expression_keys_are_passed_through_qualified() -> anyhow::Result<()> {
        let filter = LibsqlSearchFilter::eq("metadata->>'$.source'", serde_json::json!("docs"));
        let (condition, params) = filter.render()?;
        anyhow::ensure!(
            condition == "metadata->>'$.source' = ?",
            "unexpected condition: {condition}"
        );
        anyhow::ensure!(
            params == vec![libsql::Value::Text("docs".to_string())],
            "unexpected params: {params:?}"
        );
        Ok(())
    }

    #[test]
    fn and_or_combine_filters() -> anyhow::Result<()> {
        let filter = LibsqlSearchFilter::gt("priority", serde_json::json!(5))
            .and(LibsqlSearchFilter::lt("priority", serde_json::json!(20)));
        let (condition, params) = filter.render()?;
        anyhow::ensure!(
            condition == "(d.priority > ?) AND (d.priority < ?)",
            "unexpected condition: {condition}"
        );
        anyhow::ensure!(
            params == vec![libsql::Value::Integer(5), libsql::Value::Integer(20)],
            "unexpected params: {params:?}"
        );
        Ok(())
    }

    #[test]
    fn between_renders_inclusive_range() -> anyhow::Result<()> {
        let filter = LibsqlSearchFilter::between("priority", 1_i64..=10_i64);
        let (condition, params) = filter.render()?;
        anyhow::ensure!(
            condition == "d.priority BETWEEN ? AND ?",
            "unexpected condition: {condition}"
        );
        anyhow::ensure!(
            params == vec![libsql::Value::Integer(1), libsql::Value::Integer(10)],
            "unexpected params: {params:?}"
        );
        Ok(())
    }

    #[test]
    fn null_and_pattern_filters_render() -> anyhow::Result<()> {
        let filter = LibsqlSearchFilter::is_null("metadata->>'$.missing'")
            .and(LibsqlSearchFilter::like("title", "%O'Reilly%"));
        let (condition, params) = filter.render()?;
        anyhow::ensure!(
            condition == "(metadata->>'$.missing' IS NULL) AND (d.title like ?)",
            "unexpected condition: {condition}"
        );
        anyhow::ensure!(
            params == vec![libsql::Value::Text("%O'Reilly%".to_string())],
            "unexpected params: {params:?}"
        );
        Ok(())
    }

    #[test]
    fn not_filter_negates_clause() -> anyhow::Result<()> {
        let filter = LibsqlSearchFilter::eq("category", serde_json::json!("docs")).not();
        let (condition, params) = filter.render()?;
        anyhow::ensure!(
            condition == "NOT (d.category = ?)",
            "unexpected condition: {condition}"
        );
        anyhow::ensure!(
            params == vec![libsql::Value::Text("docs".to_string())],
            "unexpected params: {params:?}"
        );
        Ok(())
    }

    #[test]
    fn filter_param_coerces_json_numbers() -> anyhow::Result<()> {
        anyhow::ensure!(
            filter_param(serde_json::json!(42))? == libsql::Value::Integer(42),
            "integer filter param should coerce to Integer"
        );
        anyhow::ensure!(
            matches!(
                filter_param(serde_json::json!(1.5))?,
                libsql::Value::Real(_)
            ),
            "float filter param should coerce to Real"
        );
        anyhow::ensure!(
            filter_param(serde_json::json!(true))? == libsql::Value::Integer(1),
            "boolean filter param should coerce to Integer 1"
        );
        Ok(())
    }

    #[test]
    fn json_column_decodes_object_from_text_value() -> anyhow::Result<()> {
        let column = Column::new("metadata", "JSON");
        let value = libsql::Value::Text(r#"{"source":"docs","count":3}"#.to_string());
        let decoded = column_value_to_json(0, &column, value)?;
        anyhow::ensure!(
            decoded.get("source") == Some(&serde_json::json!("docs")),
            "JSON column should decode source field, got {decoded:?}"
        );
        anyhow::ensure!(
            decoded.get("count") == Some(&serde_json::json!(3)),
            "JSON column should decode count field, got {decoded:?}"
        );
        Ok(())
    }

    #[test]
    fn text_column_preserves_json_looking_string() -> anyhow::Result<()> {
        let column = Column::new("raw", "TEXT");
        let value = libsql::Value::Text(r#"{"a":1}"#.to_string());
        let decoded = column_value_to_json(0, &column, value)?;
        anyhow::ensure!(
            decoded == serde_json::json!(r#"{"a":1}"#),
            "TEXT column should preserve JSON-looking text, got {decoded:?}"
        );
        Ok(())
    }

    #[test]
    fn boolean_column_decodes_zero_and_one() -> anyhow::Result<()> {
        let column = Column::new("flag", "BOOLEAN");
        let zero = column_value_to_json(0, &column, libsql::Value::Integer(0))?;
        let one = column_value_to_json(0, &column, libsql::Value::Integer(1))?;
        anyhow::ensure!(
            zero == serde_json::json!(false),
            "0 should decode to false, got {zero:?}"
        );
        anyhow::ensure!(
            one == serde_json::json!(true),
            "1 should decode to true, got {one:?}"
        );
        Ok(())
    }

    // --- End-to-end round trip against a real in-memory libSQL database ---
    //
    // This exercises the native vector SQL dialect (FLOAT32(N) columns,
    // libsql_vector_idx, vector_top_k, vector_distance_cos) and the full
    // insert -> top_n -> top_n_ids path. A tiny deterministic embedding model
    // maps each keyword to a distinct basis vector so the nearest neighbor is
    // unambiguous.

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct RoundTripDoc {
        id: String,
        content: String,
    }

    impl LibsqlVectorStoreTable for RoundTripDoc {
        fn name() -> &'static str {
            "round_trip_documents"
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

    #[derive(Clone)]
    struct RoundTripModel;

    impl EmbeddingModel for RoundTripModel {
        const MAX_DOCUMENTS: usize = 16;
        type Client = ();
        fn make(_client: &Self::Client, _model: impl Into<String>, _dims: Option<usize>) -> Self {
            RoundTripModel
        }
        fn ndims(&self) -> usize {
            3
        }
        async fn embed_texts(
            &self,
            texts: impl IntoIterator<Item = String> + Send,
        ) -> Result<Vec<Embedding>, rig_core::embeddings::EmbeddingError> {
            Ok(texts
                .into_iter()
                .map(|t| Embedding {
                    document: t.clone(),
                    vec: round_trip_vec_for(&t),
                })
                .collect())
        }
    }

    fn round_trip_vec_for(text: &str) -> Vec<f64> {
        if text.contains("linglingdong") {
            vec![0.0, 0.0, 1.0]
        } else if text.contains("glarb") {
            vec![0.0, 1.0, 0.0]
        } else {
            vec![1.0, 0.0, 0.0]
        }
    }

    #[tokio::test]
    async fn round_trip_insert_and_top_n_with_local_libsql() -> anyhow::Result<()> {
        let db = libsql::Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        let model = RoundTripModel;

        let store: LibsqlVectorStore<_, RoundTripDoc> =
            LibsqlVectorStore::with_distance_metric(conn, &model, LibsqlDistanceMetric::Cosine)
                .await?;

        let documents: Vec<(RoundTripDoc, Vec<Embedding>)> = vec![
            (
                RoundTripDoc {
                    id: "doc0".to_string(),
                    content: "flurbo".to_string(),
                },
                vec![Embedding {
                    document: "flurbo".to_string(),
                    vec: round_trip_vec_for("flurbo"),
                }],
            ),
            (
                RoundTripDoc {
                    id: "doc1".to_string(),
                    content: "glarb".to_string(),
                },
                vec![Embedding {
                    document: "glarb".to_string(),
                    vec: round_trip_vec_for("glarb"),
                }],
            ),
            (
                RoundTripDoc {
                    id: "doc2".to_string(),
                    content: "linglingdong".to_string(),
                },
                vec![Embedding {
                    document: "linglingdong".to_string(),
                    vec: round_trip_vec_for("linglingdong"),
                }],
            ),
        ];

        store.add_rows(documents).await?;

        let index = store.index(model);

        let req = VectorSearchRequest::builder()
            .query("What is a linglingdong?")
            .samples(1)
            .build();

        let results = index.top_n::<RoundTripDoc>(req.clone()).await?;
        anyhow::ensure!(
            results.len() == 1,
            "expected 1 top_n result, got {}",
            results.len()
        );
        let (score, id, _doc) = results
            .first()
            .ok_or_else(|| anyhow::anyhow!("top_n returned no results"))?;
        anyhow::ensure!(id == "doc2", "expected doc2 as nearest, got {id}");
        anyhow::ensure!(
            *score > 0.99,
            "cosine similarity for identical vectors should be ~1.0, got {score}"
        );

        let id_results = index.top_n_ids(req).await?;
        anyhow::ensure!(
            id_results.len() == 1,
            "expected 1 top_n_ids result, got {}",
            id_results.len()
        );
        let (id_score, id_id) = id_results
            .first()
            .ok_or_else(|| anyhow::anyhow!("top_n_ids returned no results"))?;
        anyhow::ensure!(id_id == "doc2", "expected doc2 id, got {id_id}");
        anyhow::ensure!(
            id_score == score,
            "top_n and top_n_ids scores should match, got {id_score} vs {score}"
        );

        Ok(())
    }

    #[tokio::test]
    async fn round_trip_filter_and_threshold_with_local_libsql() -> anyhow::Result<()> {
        let db = libsql::Builder::new_local(":memory:").build().await?;
        let conn = db.connect()?;
        let model = RoundTripModel;

        let store: LibsqlVectorStore<_, RoundTripDoc> =
            LibsqlVectorStore::new(conn, &model).await?;

        let documents: Vec<(RoundTripDoc, Vec<Embedding>)> = vec![
            (
                RoundTripDoc {
                    id: "a".to_string(),
                    content: "flurbo".to_string(),
                },
                vec![Embedding {
                    document: "flurbo".to_string(),
                    vec: round_trip_vec_for("flurbo"),
                }],
            ),
            (
                RoundTripDoc {
                    id: "b".to_string(),
                    content: "linglingdong".to_string(),
                },
                vec![Embedding {
                    document: "linglingdong".to_string(),
                    vec: round_trip_vec_for("linglingdong"),
                }],
            ),
        ];
        store.add_rows(documents).await?;

        let index = store.index(model);

        // A threshold just above 1.0 - epsilon should filter out everything
        // because no stored vector matches the "glarb" query exactly.
        let req = VectorSearchRequest::builder()
            .query("glarb")
            .samples(5)
            .threshold(0.99)
            .build();
        let results = index.top_n::<RoundTripDoc>(req).await?;
        anyhow::ensure!(
            results.is_empty(),
            "threshold should drop all non-matching candidates, got {results:?}"
        );

        // A filter on the content column should restrict to the matching doc.
        let req = VectorSearchRequest::builder()
            .query("linglingdong")
            .samples(5)
            .filter(LibsqlSearchFilter::eq(
                "content",
                serde_json::json!("linglingdong"),
            ))
            .build();
        let results = index.top_n::<RoundTripDoc>(req).await?;
        anyhow::ensure!(
            results.len() == 1,
            "content filter should leave exactly one result, got {}",
            results.len()
        );
        anyhow::ensure!(
            results.first().is_some_and(|(_, id, _)| id == "b"),
            "filtered result should be doc b"
        );

        Ok(())
    }
}
