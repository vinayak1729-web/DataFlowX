import streamlit as st
import pandas as pd
from google.cloud import storage, bigquery
from google.adk.runtime import run_agent
import os
from dotenv import load_dotenv
import json

load_dotenv()

# Initialize clients
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
storage_client = storage.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))
bq_client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

# Configuration
BUCKET_NAME = "gdgadk-bucket"
DATASET_ID = "mcp_dataset"
TABLE_ID = "uploaded_data"

def upload_to_gcs(file, bucket_name, blob_name):
    """Upload file to Google Cloud Storage."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_file(file)
    return f"gs://{bucket_name}/{blob_name}"

def load_to_bigquery(gcs_uri, dataset_id, table_id):
    """Load CSV from GCS to BigQuery."""
    dataset_ref = bq_client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
        write_disposition=bigquery.WriteTruncate
    )
    load_job = bq_client.load_table_from_uri(gcs_uri, table_ref, job_config=job_config)
    load_job.result()  # Wait for job to complete
    return f"Loaded {load_job.output_rows} rows to {dataset_id}.{table_id}"

def trigger_adk_agent(query):
    """Trigger ADK agent and get response."""
    agent_response = run_agent(
        agent_path="adk_agent",
        query=query,
        model="gemini-2.0-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    return agent_response

# Streamlit UI
st.title("CSV to BigQuery with ADK Agent")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Upload to GCS
    gcs_uri = upload_to_gcs(uploaded_file, BUCKET_NAME, "uploaded_data.csv")
    st.success(f"Uploaded CSV to {gcs_uri}")

    # Load to BigQuery
    try:
        result = load_to_bigquery(gcs_uri, DATASET_ID, TABLE_ID)
        st.success(result)
    except Exception as e:
        st.error(f"Error loading to BigQuery: {str(e)}")
        st.stop()

    # Query input for ADK agent
    query = st.text_input("Ask a question about the data (e.g., 'Show top 5 rows from mcp_dataset.uploaded_data')")
    if query:
        try:
            response = trigger_adk_agent(query)
            st.write("Agent Response:")
            st.json(json.loads(response))  # Display response as JSON for clarity
        except Exception as e:
            st.error(f"Error querying agent: {str(e)}")

if __name__ == "__main__":
    st.write("Ready to upload CSV and query data!")