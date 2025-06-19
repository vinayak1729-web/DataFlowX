from flask import Flask, jsonify, request
from google.cloud import bigquery
from google.cloud.exceptions import NotFound
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# Set up BigQuery client with service account credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
client = bigquery.Client(project=os.getenv("GOOGLE_CLOUD_PROJECT"))

@app.route("/tools/list_tables", methods=["POST"])
def list_tables_endpoint():
    """List all tables in a specified BigQuery dataset."""
    dataset_id = request.json.get("dataset_id")
    try:
        dataset_ref = client.dataset(dataset_id)
        tables = list(client.list_tables(dataset_ref))
        table_names = [table.table_id for table in tables]
        return jsonify({"status": "success", "tables": table_names})
    except NotFound:
        return jsonify({"status": "error", "message": f"Dataset '{dataset_id}' not found."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/tools/query_table", methods=["POST"])
def query_table_endpoint():
    """Run a SQL query on a specified BigQuery table."""
    data = request.json
    dataset_id = data.get("dataset_id")
    table_id = data.get("table_id")
    query = data.get("query")
    try:
        full_table_id = f"{os.getenv('GOOGLE_CLOUD_PROJECT')}.{dataset_id}.{table_id}"
        safe_query = query.replace(full_table_id, f"`{full_table_id}`")
        query_job = client.query(safe_query)
        results = query_job.result()
        rows = [dict(row) for row in results]
        return jsonify({"status": "success", "results": rows})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/tools", methods=["GET"])
def list_tools():
    """List available tools."""
    return jsonify(["list_tables", "query_table"])

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)