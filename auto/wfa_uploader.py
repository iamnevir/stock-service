import os
import io
import sys
from bson import ObjectId
import pandas as pd
from typing import List, Dict
import traceback
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pymongo import MongoClient

from auto.utils import get_mongo_uri
from auto.wfa import import_wfa
import pandas as pd

def drop_sparse_columns(
    df: pd.DataFrame,
    min_ratio: float = 0.8,
    treat_zero_as_nan: bool = False
) -> pd.DataFrame:
    """
    Loại bỏ các cột có ít dữ liệu

    min_ratio: tỷ lệ tối thiểu số row có data (0.0 - 1.0)
    treat_zero_as_nan: coi 0 là missing hay không
    """

    df = df.copy()
    total_rows = len(df)

    if total_rows == 0:
        return df

    if treat_zero_as_nan:
        df = df.replace(0, pd.NA)

    # count non-null per column
    non_null_ratio = df.notna().sum() / total_rows

    # giữ lại cột đủ data
    valid_cols = non_null_ratio[non_null_ratio >= min_ratio].index

    return df[valid_cols]

def get_sheets_service():
    SCOPES = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets.readonly"
    ]

    CREDENTIAL_FILE = "/home/ubuntu/nevir/credentials.json"
    TOKEN_FILE = "/home/ubuntu/nevir/token.json"

    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIAL_FILE, SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())

    return build("sheets", "v4", credentials=creds)


def extract_sheet_id(url: str) -> str:
    return url.split("/d/")[1].split("/")[0]



def load_sheet_to_df(spreadsheet_id: str) -> pd.DataFrame:
    service = get_sheets_service()

    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range="A1:ZZZ",
        majorDimension="ROWS"
    ).execute()

    values = result.get("values", [])
    if not values:
        return pd.DataFrame()

    raw_headers = values[0]
    rows = values[1:]

    # Lấy index của các cột có header
    valid_col_indexes = [
        i for i, h in enumerate(raw_headers) if str(h).strip() != ""
    ]

    headers = [raw_headers[i] for i in valid_col_indexes]

    cleaned_rows = []
    for row in rows:
        cleaned_rows.append([
            row[i] if i < len(row) else ""
            for i in valid_col_indexes
        ])

    return pd.DataFrame(cleaned_rows, columns=headers)



def import_wfa_from_links(alpha_id):
    mongo_client = MongoClient(get_mongo_uri())
    alpha_db = mongo_client["alpha"]
    alpha_collection = alpha_db["alpha_collection"]

    alpha_doc = alpha_collection.find_one({"_id": ObjectId(alpha_id)})
    if not alpha_doc:
        raise ValueError("Không tìm thấy alpha_collection với id này.")

    wfa = alpha_doc.get("wfa", [])

    # chỉ lấy item chưa done
    wfa_items = [
        item for item in wfa
        if item.get("status") != "done"
    ]
    for item in wfa_items:
        link = item.get("link", None)
        is_data = item.get("is", {})

        try:
            # 1️⃣ Link trống
            if not link:
                alpha_collection.update_one(
                    {"_id": ObjectId(alpha_id), "wfa.link": link},
                    {"$set": {
                        "wfa.$.status": "error",
                        "wfa.$.message": "Link trống",
                    }}
                )
                continue

            # 2️⃣ Load sheet
            try:
                sid = extract_sheet_id(link)
                df = load_sheet_to_df(sid)
                df = df.loc[:, ~df.columns.duplicated()]
                for col in df.columns:
                    if col != "Strategy":
                        df[col] = pd.to_numeric(df[col], errors="ignore")
            except Exception as e:
                print(f"Error loading sheet from link {link}: {e}")
                alpha_collection.update_one(
                    {"_id": ObjectId(alpha_id), "wfa.link": link},
                    {"$set": {
                        "wfa.$.status": "error",
                        "wfa.$.message": "Không đọc được Google Sheet",
                        "wfa.$.error": str(e),
                    }}
                )
                continue

            # 3️⃣ Sheet rỗng
            if df.empty:
                alpha_collection.update_one(
                    {"_id": ObjectId(alpha_id), "wfa.link": link},
                    {"$set": {
                        "wfa.$.status": "error",
                        "wfa.$.message": "Sheet rỗng",
                        "wfa.$.error": "Empty sheet",
                    }}
                )
                continue

            # 4️⃣ Import
            result = import_wfa(
                df=df,
                id=alpha_id,
                start=is_data.get("start", ""),
                end=is_data.get("end", ""),
            )
            if not result:
                alpha_collection.update_one(
                    {"_id": ObjectId(alpha_id), "wfa.link": link},
                    {"$set": {
                        "wfa.$.status": "error",
                        "wfa.$.message": "Import thất bại",
                        "wfa.$.error": "Import function returned False",
                    }}
                )
                continue

            # 5️⃣ Thành công
            alpha_collection.update_one(
                {"_id": ObjectId(alpha_id), "wfa.link": link},
                {"$set": {
                    "wfa.$.status": "done",
                    "wfa.$.message": "Import thành công",
                }}
            )

        except Exception as e:
            alpha_collection.update_one(
                {"_id": ObjectId(alpha_id), "wfa.link": link},
                {"$set": {
                    "wfa.$.status": "error",
                    "wfa.$.message": "Import thất bại",
                    "wfa.$.error": traceback.format_exc(),
                }}
            )
    alpha_collection.update_one(
        {"_id": ObjectId(alpha_id)},
        {"$set": {
            "wfa_importing": False
        }}
    )

def main():
    if len(sys.argv) < 2:
        print("Usage: /home/ubuntu/anaconda3/bin/python /home/ubuntu/nevir/auto/wfa_uploader.py <_id>")
        sys.exit(1)

    _id = sys.argv[1]

    import_wfa_from_links(_id)

if __name__ == "__main__":
    main()
