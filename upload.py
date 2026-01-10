import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

def get_drive_service():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    CREDENTIAL_FILE = "/home/ubuntu/nevir/credentials.json"
    creds = None
    if os.path.exists('/home/ubuntu/nevir/token.json'):
        creds = Credentials.from_authorized_user_file('/home/ubuntu/nevir/token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIAL_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open('/home/ubuntu/nevir/token.json', 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)


def upload_file_to_drive(save_file):
    try:
        sheet_name = os.path.splitext(os.path.basename(save_file))[0]
        media = MediaFileUpload(save_file, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        file_metadata = {
            'name': sheet_name,
            'mimeType': 'application/vnd.google-apps.spreadsheet',
            'parents': ["1ZI65HWxDFaPcXQYyJFdu47cLQtwJ_4Iw"]
        }
        service = get_drive_service()
        uploaded = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        file_id = uploaded.get('id')
        sheet_url = f"https://docs.google.com/spreadsheets/d/{file_id}"
        print(f"   📤 Uploaded to Google Drive: {sheet_url}")
    except Exception as e:
        print(f"   ❌ Upload failed: {e}")


for file_name in os.listdir('/home/ubuntu/nevir/gen_spot/results_1_2'):
    if file_name.endswith('.xlsx'):
        full_path = os.path.join('/home/ubuntu/nevir/gen_spot/results_1_2', file_name)
        upload_file_to_drive(full_path)
        