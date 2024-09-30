import time
import pyodbc
from ftplib import FTP
from dotenv import load_dotenv
from datetime import datetime
from decimal import Decimal
from google.cloud import speech
from google.cloud import storage
import os
import google.generativeai as genai

from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build
from pydub import AudioSegment
import yt_dlp
import anthropic
from langdetect import detect_langs, lang_detect_exception
import re
import logging
from logging.handlers import RotatingFileHandler

from langdetect import detect

# Load environment variables
load_dotenv()

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
API_KEY = os.getenv("API_KEY")
CONNECTION_STRING = os.getenv("CONNECTION_STRING")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
speech_client = speech.SpeechClient()

# TODO:  Need to add language prefix to WAV file 


# Setup logging with a rotating file handler to limit log file size
log_file = "app.log"
log_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=3)  # 5MB per file, 3 backups
log_handler.setLevel(logging.INFO)

# Log format
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
log_handler.setFormatter(formatter)

# Setup logging to both rotating file and console
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        log_handler,
        logging.StreamHandler()  # For logging to console
    ]
)

# Setup logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

def detect_language_from_title(video_id):
    """
    Detect the language of a YouTube video based on its title.
    """
    try:
        youtube = build("youtube", "v3", developerKey=API_KEY)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        if response['items']:
            video_title = response['items'][0]['snippet']['title']
            logging.info(f"Detected title: {video_title}")
            detected_lang = detect(video_title)
            logging.info(f"Detected language for video title: {detected_lang}")
            return detected_lang
        else:
            logging.warning(f"No video found with video ID: {video_id}")
            return 'en'  # Default to English if unable to detect
    except Exception as e:
        logging.error(f"Error detecting language from title: {e}")
        return 'en'  # Default to English on error

def get_db_connection():
    try:
        connection = pyodbc.connect(CONNECTION_STRING)
        # logging.info("Successfully connected to the database.")
        return connection
    except pyodbc.Error as e:
        logging.error(f"Error connecting to SQL Server: {e}")
        return None

def execute_query(query, params=()):
    conn = get_db_connection()
    if not conn:
        logging.error("Database connection failed.")
        return []
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            results = cursor.fetchall()
            logging.info(f"Executed query: {query} with params: {params}")
            return results
    except pyodbc.Error as e:
        logging.error(f"Error executing query: {e}")
        return []
    finally:
        conn.close()

def fetch_indexes():
    query = """
    SELECT TOP (1) [indexNoPk], [LockQ], [type], [username], [gComment], [produc], [CustName]
    FROM [DB_164462_kdsisdlo291].[dbo].[Quote]
    WHERE [produc] = 'Pending'
    """
    return execute_query(query)

def get_yt(index_no_pk):
    query = """
    SELECT TOP (1) [Address1], [City], [State]
    FROM [DB_164462_kdsisdlo291].[dbo].[ClientesRCHome]
    WHERE [indexNoPk] = ?
    """
    return execute_query(query, (index_no_pk,))

def fetch_client(user_q):
    query = """
    SELECT [userQ], [name], [salesID], [salesIdGP], [mail1]
    FROM [UserQuote]
    WHERE [userQ] = ?
    """
    return execute_query(query, (user_q,))

def check_file_existence(bucket_name, file_name):
    try:
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(file_name)
        exists = blob.exists()
        logging.info(f"Checked existence for {file_name} in bucket {bucket_name}: {exists}")
        return exists
    except Exception as e:
        logging.error(f"Error checking file existence in bucket: {e}")
        return False

def read_file_from_bucket(bucket_name, blob_name):
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_string()
        content_str = content.decode("utf-8")
        logging.info(f"Downloaded file {blob_name} from bucket {bucket_name}.")
        return content_str
    except Exception as e:
        logging.error(f"Error reading file from bucket: {e}")
        return ""

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    try:
        destination_blob_name = destination_blob_name.replace("_trim", "")
        if not source_file_name.startswith("downloads/"):
            source_file_name = os.path.join("downloads", source_file_name)

        if not os.path.isfile(source_file_name):
            raise FileNotFoundError(f"File not found: {source_file_name}")

        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        logging.info(f"File {source_file_name} uploaded to {destination_blob_name}.")
    except Exception as e:
        logging.error(f"Error uploading blob: {e}")

def get_youtube_transcript(video_id):
    """
    Get YouTube transcript using the correct language code based on video title.
    """
    try:
        # Detect language of the video title
        detected_language = detect_language_from_title(video_id)

        # Attempt to retrieve the transcript in the detected language
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[detected_language])
        full_transcript = " ".join([entry['text'] for entry in transcript])

        logging.info(f"Retrieved YouTube transcript for video ID {video_id} in language: {detected_language}.")
        return full_transcript
    except Exception as e:
        logging.error(f"Error retrieving YouTube transcript: {e}")
        return None

def retrieve_or_generate_transcript(video_id, start_time=0, end_time=0):
    try:
        # Set the bucket name and transcript filename
        bucket_name = "spchtotxt1stop"  # Replace with your bucket name
        txt_file_name = f"{video_id}.txt"
        
        logging.info(f"Checking if transcript file {txt_file_name} exists in Google Cloud bucket {bucket_name}.")
        
        # Check if the file exists in the Google Cloud bucket
        file_exists = check_file_existence(bucket_name, txt_file_name)
        logging.info(f"Transcript {txt_file_name} existence check returned: {file_exists}")
        
        # If the file exists, download and return the transcript
        if file_exists:
            logging.info(f"Transcript {txt_file_name} found in Google Cloud. Attempting to download...")
            transcript = read_file_from_bucket(bucket_name, txt_file_name)
            if transcript:
                logging.info(f"Successfully downloaded transcript {txt_file_name} from Google Cloud.")
                return transcript
            else:
                logging.error(f"Failed to download transcript {txt_file_name} from Google Cloud.")
                return None
        else:
            logging.info(f"Transcript {txt_file_name} not found in Google Cloud. Proceeding with transcript generation.")
        
        # Check for captions or process audio if no file exists in the bucket
        captions = check_captions(video_id, API_KEY)
        if captions == "Captions are available.":
            logging.info(f"Captions available for video ID {video_id}, retrieving YouTube transcript.")
            transcript = get_youtube_transcript(video_id)
            if transcript:
                logging.info("Successfully retrieved captions from YouTube.")
                save_transcript_to_google_cloud(bucket_name, txt_file_name, transcript)
                return transcript
            else:
                logging.error("Failed to retrieve captions from YouTube.")
                return None
        else:
            logging.info("No captions available, downloading and processing audio.")
            filename_wav = download_audio(video_id, start_time, end_time)
            transcript = process_audio(filename_wav)
            if transcript:
                logging.info("Successfully processed audio for transcript.")
                save_transcript_to_google_cloud(bucket_name, txt_file_name, transcript)
                return transcript
            else:
                logging.error("Failed to process audio for transcript.")
                return None
    except Exception as e:
        logging.error(f"Error retrieving or generating transcript for video ID {video_id}: {e}")
        return None

def save_transcript_to_google_cloud(bucket_name, txt_file_name, transcript):
    try:
        logging.info(f"Saving transcript {txt_file_name} to Google Cloud bucket {bucket_name}.")
        
        # Save the transcript as a .txt file locally first
        local_file_path = os.path.join("downloads", txt_file_name)
        with open(local_file_path, "w", encoding="utf-8") as file:
            file.write(transcript)
        
        # Upload the transcript to Google Cloud
        upload_blob(bucket_name, local_file_path, txt_file_name)
        logging.info(f"Transcript {txt_file_name} successfully uploaded to Google Cloud.")
    except Exception as e:
        logging.error(f"Error saving transcript {txt_file_name} to Google Cloud: {e}")

def process_audio(filename):
    try:
        config = speech.RecognitionConfig(
            sample_rate_hertz=16000,
            language_code="en-US",
            audio_channel_count=1,
        )
        media_uri = f"gs://spchtotxt1stop/{filename}"
        audio = speech.RecognitionAudio(uri=media_uri)
        operation = speech_client.long_running_recognize(config=config, audio=audio)
        response = operation.result(timeout=180)  # Increased timeout for longer audio
        transcript = " ".join([result.alternatives[0].transcript for result in response.results])
        logging.info(f"Processed audio file {filename} and obtained transcript.")
        return transcript
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
        return None

def download_audio(video_id, start_mark, end_mark):
    try:
        logging.info(f"Starting download for video ID {video_id}, Start: {start_mark} mins, End: {end_mark} mins.")
        output_folder = "downloads"
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
            "outtmpl": f"{output_folder}/{video_id}.%(ext)s",
        }

        # Step 1: Download the audio file from YouTube
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=True)
            output_filename = ydl.prepare_filename(info_dict)

            if not output_filename.endswith(".wav"):
                output_filename = output_filename.replace(".webm", ".wav").replace(".m4a", ".wav")

            output_filename = os.path.join(output_folder, os.path.basename(output_filename))

            if not os.path.isfile(output_filename):
                raise FileNotFoundError(f"Expected WAV file not found: {output_filename}")

        # Step 2: Convert start_mark and end_mark to integers
        start_mark = int(start_mark)
        end_mark = int(end_mark)

        # Step 3: Load the audio file
        sound = AudioSegment.from_wav(output_filename)
        
        # Step 4: Convert start_mark and end_mark from minutes to milliseconds
        start_ms = start_mark * 60 * 1000
        if end_mark == 0:
            end_ms = len(sound)  # Use the full duration of the audio if no end mark is specified
        else:
            end_ms = end_mark * 60 * 1000
        
        # Step 5: Clip the audio between start and end marks
        clipped_sound = sound[start_ms:end_ms].set_channels(1)
        
        # Step 6: Export the clipped audio to a new file
        clipped_output_filename = os.path.join(output_folder, f"{video_id}_clipped.wav")
        clipped_sound.export(clipped_output_filename, format="wav")
        logging.info(f"Downloaded and clipped audio file saved to {clipped_output_filename}.")
        
        return clipped_output_filename
    except Exception as e:
        logging.error(f"Error downloading or clipping audio: {e}")
        return ""

def check_captions(video_id, api_key):
    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.captions().list(part="snippet", videoId=video_id)
        response = request.execute()

        if response.get("items"):
            logging.info(f"Captions are available for video ID {video_id}.")
            return "Captions are available."
        else:
            logging.info(f"No captions available for video ID {video_id}.")
            return "No captions available."
    except Exception as e:
        logging.error(f"Error checking captions for video ID {video_id}: {e}")
        return "Error checking captions."

def ftp_upload(local_filepath, filename):
    try:
        ftp = FTP("ftp.1stopbot.com")
        ftp.login(user="1stopbot", passwd="ks9sJDasdd")
        ftp.cwd("/1stopbot/wwwroot/clientFiles/")
        with open(local_filepath, "rb") as file:
            ftp.storbinary(f"STOR {filename}", file)
        ftp.quit()
        logging.info(f"File {filename} uploaded successfully via FTP.")
    except Exception as e:
        logging.error(f"Error uploading file to FTP: {e}")

def show_user_accessible_page_link():
    if is_user_logged_in():
        user_id = get_current_user_id()
        membership_level = pmpro_getMembershipLevelForUser(user_id)

        logging.info(f"User ID: {user_id}, Membership Level: {membership_level.name if membership_level else 'None'}")

        if membership_level:
            if membership_level.name == 'ZIP33166':
                link = f'<a href="{home_url("/zip33166-page/")}">Access ZIP 33166 Page</a>'
                logging.info("Providing link for ZIP33166 membership.")
                return link
            elif membership_level.name == 'ZIP33014':
                link = f'<a href="{home_url("/zip33014-page/")}">Access ZIP 33014 Page</a>'
                logging.info("Providing link for ZIP33014 membership.")
                return link
            else:
                logging.info("User has a membership but no specific page link is assigned.")
                return '<p>You do not have access to any special pages. Please contact support.</p>'
        else:
            logging.info("User does not have an active membership.")
            return '<p>You do not have an active membership.</p>'
    else:
        logging.info("User is not logged in.")
        return '<p>Please <a href="' + wp_login_url() + '">log in</a> to access your page.</p>'

def display_accessible_page_link_shortcode():
    return show_user_accessible_page_link()

def insert_anal(lock_q, rating, desc, item):
    try:
        logging.info(f"Inserting analysis record: LockQ={lock_q}, Rating={rating}, Desc={desc}, Item={item}")
        connection = pyodbc.connect(CONNECTION_STRING)
        cursor = connection.cursor()
        desc = sanitize_input(desc[:50])
        sql_query = """
            INSERT INTO [DB_164462_kdsisdlo291].[dbo].[Analysis] 
            (indexFk, itemRating, itemDesc, item, itemCat) 
            VALUES (?, ?, ?, ?, '-')
        """
        cursor.execute(sql_query, (str(lock_q), str(rating), str(desc), str(item)))
        connection.commit()
        cursor.close()
        connection.close()
        logging.info("Analysis record inserted successfully.")
    except pyodbc.Error as e:
        logging.error(f"Error inserting analysis record: {e}")

def sanitize_input(input_string):
    if input_string is None:
        return input_string
    sanitized = input_string.replace("'", "''").replace('"', '').replace('\\', '')
    return sanitized

def check_and_load_transcript(file_path, google_bucket, txt_storage_file):
    """
    Checks if the transcript exists locally or in Google Cloud, and loads it.
    
    Parameters:
    - file_path (str): The local file path where the transcript is expected.
    - google_bucket (str): The name of the Google Cloud Storage bucket.
    - txt_storage_file (str): The filename in the Google Cloud bucket.
    
    Returns:
    - str: The transcript content if found, None otherwise.
    """
    try:
        # Check if the transcript file exists locally
        if os.path.exists(file_path):
            logging.info(f"Transcript found locally at {file_path}.")
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()

        # If not found locally, check in the Google Cloud bucket
        elif check_file_existence(google_bucket, txt_storage_file):
            logging.info(f"Transcript found in Google Cloud bucket {google_bucket}. Downloading...")
            transcript_content = read_file_from_bucket(google_bucket, txt_storage_file)
            if transcript_content:
                # Optionally, save it locally for future use
                with open(file_path, "w", encoding="utf-8") as file:
                    file.write(transcript_content)
                logging.info(f"Transcript saved locally at {file_path} after downloading from Google Cloud.")
                return transcript_content
            else:
                logging.error(f"Transcript found in Google Cloud but failed to download.")
                return None
        else:
            logging.info("Transcript not found locally or in Google Cloud.")
            return None
    except Exception as e:
        logging.error(f"Error in check_and_load_transcript: {e}")
        return None

def update_transcript_record(file_path, clientes_index_no):
    """
    Updates the database record where the 'type' field is 'Transcript' for a given clientes_index_no,
    marking the transcript as 'Completed' and updating the gComment field with the transcript filename.
    
    Args:
        file_path (str): The path or filename of the transcript to be stored in the gComment field.
        clientes_index_no (int): The index number of the ClientesRCHome record in the database.
    """
    try:
        connection = pyodbc.connect(CONNECTION_STRING)
        cursor = connection.cursor()
        
        # SQL query with a filter on 'type' to ensure only 'Transcript' types are updated
        update_query = """
            UPDATE [DB_164462_kdsisdlo291].[dbo].[Quote]
            SET [produc] = 'Completed', [gComment] = ?
            WHERE [LockQ] = ? AND [type] = 'Transcript'
        """
        
        # Output the query and parameters for debugging
        query_for_debug = f"""
            UPDATE [DB_164462_kdsisdlo291].[dbo].[Quote]
            SET [produc] = 'Completed', [gComment] = '{file_path}'
            WHERE [indexNoPk] = {clientes_index_no} AND [type] = 'Transcript'
        """
        #print(f"Generated SQL Query: {query_for_debug}")
        logging.info(f"Generated SQL Query: {query_for_debug}")
        
        # Executing the update query
        cursor.execute(update_query, (file_path, clientes_index_no))
        
        # Commit the transaction
        connection.commit()
        print(f"Transcript record updated for clientes_index_no {clientes_index_no} with file {file_path}.")
        logging.info(f"Transcript record updated for clientes_index_no {clientes_index_no} with file {file_path}.")
        
    except pyodbc.Error as e:
        print(f"Error updating transcript record for clientes_index_no {clientes_index_no}: {e}")
        logging.error(f"Error updating transcript record for clientes_index_no {clientes_index_no}: {e}")
    finally:
        # Ensure that the cursor and connection are properly closed
        cursor.close()
        connection.close()



def update_quote(file_path, index):
    connection = pyodbc.connect(CONNECTION_STRING)
    # Initialise the Cursor
    cursor = connection.cursor()
    # Executing a SQL Query
    cursor.execute(
        "UPDATE [DB_164462_kdsisdlo291].[dbo].[Quote] SET [produc] = 'Completed', [gComment] = '"
        + file_path
        + "' WHERE [indexNoPk] = "
        + str(index)
    )
    # resultId = cursor.fetchval()
    # print(f"Inserted Product ID : {resultId}")
    connection.commit()
    cursor.close()
    connection.close()

def select_prompt_tbl(resource_type, language):
    """
    Retrieves the AI prompt from the database based on the resource type and language.
    
    Parameters:
    - resource_type (str): The type of resource (e.g., 'Transcript', 'Summary').
    - language (str): The language code (e.g., 'en' for English, 'pt' for Portuguese).
    
    Returns:
    - str: The prompt text, or "ERROR FINDING PROMPT" if the prompt could not be found.
    """
    query = """
        SELECT TOP (1) [promptName], [promptText] 
        FROM [DB_164462_kdsisdlo291].[dbo].[prompt] 
        WHERE [promptName] = ?
    """
    
    # Execute the query with the resource_type as the parameter
    results = execute_query(query, (resource_type,))
    
    if results:
        # Replace the {language} placeholder with the actual detected language
        prompt_use = results[0][1].replace("{language}", language)
        return prompt_use
    else:
        # Return an error if no prompt is found
        logging.error(f"Error finding prompt for resource_type '{resource_type}' and language '{language}'")
        return "ERROR FINDING PROMPT"

from langdetect import detect_langs, lang_detect_exception

def detect_full_language(text):
    try:
        # Detect the language using langdetect
        detector = detect_langs(text)
        detected_lang_code = detector[0].lang
        
        # Map language codes to full language names
        lang_map = {
            "en": "English",
            "es": "Spanish",
            "pt": "Portuguese",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            # Add more language mappings as needed
        }
        
        detected_lang = lang_map.get(detected_lang_code, "Unknown")
        return detected_lang
    except lang_detect_exception.LangDetectException:
        return "Language detection failed"


def generate_gemini_content(transcript_text, prompt1):
    try:
        if transcript_text is None:
            print("Transcript was empty when running Gemini")
            return None  # Exit if transcript is empty
        if prompt1 is None:
            print("Prompt was empty when accessing Gemini")
            return None  # Exit if prompt is empty

        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt1 + transcript_text)
        return response.text

    except Exception as e:
        # Catch any exception and print the error message
        print(f"An error occurred: {str(e)}")
        return None  # Return None if an error occurs

def ftp_upload(local_filepath, filename):
    try:
        ftp = FTP("ftp.1stopbot.com")
        ftp.login(user="1stopbot", passwd="ks9sJDasdd")
        ftp.cwd("/1stopbot/wwwroot/clientFiles/")
        with open(local_filepath, "rb") as file:
            ftp.storbinary(f"STOR {filename}", file)
        ftp.quit()
        print(f"File {filename} uploaded successfully.")
    except Exception as e:
        print(f"Error uploading file to FTP: {e}")

def execute_update(query, params=()):
    conn = get_db_connection()
    if not conn:
        logging.error("Database connection failed.")
        return
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params)
            conn.commit()  # Commit the update
            logging.info(f"Executed update: {query} with params: {params}")
    except pyodbc.Error as e:
        logging.error(f"Error executing update: {e}")
    finally:
        conn.close()


def run_periodically(interval):
    try:
        print("Step 1: Starting the periodic processing loop.")
        logging.info("Step 1: Starting the periodic processing loop.")
        
        while True:
            print("Step 2: Retrieving the next pending record.")
            logging.info("Step 2: Retrieving the next pending record.")
            quote_indexes = fetch_indexes()

            if quote_indexes:
                print("Step 3: Records found for processing.")
                logging.info("Step 3: Records found for processing.")
                
                for item in quote_indexes:
                    quote_index_no = item[0]
                    clientes_index_no = item[1]
                    type_info = item[2]
                    username = item[3]
                    g_comment = item[4]
                    produc_val = item[5]
                    cust_name = item[6]

                    print(f"Step 4: Processing Quote Index No: {quote_index_no}")
                    logging.info(f"Step 4: Processing Quote Index No: {quote_index_no}")
                    
                    print(f"Step 5: Clientes Index No: {clientes_index_no}")
                    logging.info(f"Step 5: Clientes Index No: {clientes_index_no}")

                    #-------------------------
                    # Step 5a: Check if the transcript already exists locally
                    quote_records = get_yt(clientes_index_no)  # Fetch YouTube address from ClientesRCHome table

                    if quote_records:
                        yt_address = quote_records[0][0]  # Address1 field holds the YouTube video link
                        start_mark = quote_records[0][1]  # City field holds the start time of the video segment
                        end_mark = quote_records[0][2]    # State field holds the end time of the video segment

                        # Extract video ID from the YouTube address
                        video_id = extract_video_id(yt_address)
                        txt_storage_file = f"{video_id}.txt"
                        file_path = "clientAssets/"
                        full_path_transc = os.path.join(file_path, txt_storage_file)

                        # Ensure full_path_transc is defined and exists
                        if full_path_transc and os.path.exists(full_path_transc):
                            print(f"Step 5a: Transcript already exists locally at {full_path_transc}")
                            logging.info(f"Step 5a: Transcript already exists locally at {full_path_transc}")

                            # Read the transcript text file and load the data into the transcript_text variable
                            with open(full_path_transc, "r", encoding="utf-8") as file:
                                transcript_text = file.read()
                            print(f"Step 5a: Transcript text successfully loaded from {full_path_transc}")
                            logging.info(f"Step 5a: Transcript text successfully loaded from {full_path_transc}")

                            # Query the database to check if the 'produc' field is set to 'Pending'
                            query = """
                                SELECT [produc]
                                FROM [DB_164462_kdsisdlo291].[dbo].[Quote]
                                WHERE [LockQ] = ? AND [type] = 'Transcript'
                            """
                            result = execute_query(query, (clientes_index_no,))

                            if result and result[0][0].rstrip() == 'Pending':
                                print(f"Step 5a: Record found with 'Pending' status. Proceeding to mark as 'Completed'.")
                                logging.info(f"Step 5a: Record found with 'Pending' status. Proceeding to mark as 'Completed'.")

                                # Update the transcript record in the database using LockQ
                                update_transcript_record(txt_storage_file, clientes_index_no)

                                # Upload the transcript to the FTP server
                                try:
                                    ftp_upload(full_path_transc, txt_storage_file)
                                    print(f"Step 5a: Transcript file '{txt_storage_file}' uploaded to FTP.")
                                    logging.info(f"Step 5a: Transcript file '{txt_storage_file}' uploaded to FTP.")
                                except Exception as e:
                                    print(f"Step 5a: Error uploading transcript file: {e}")
                                    logging.error(f"Step 5a: Error uploading transcript file: {e}")

                                # Proceed with further processing (like calling AI) instead of skipping
                                print("Step 5a: Proceeding to Step 10 to process the transcript.")
                                logging.info("Step 5a: Proceeding to Step 10 to process the transcript.")
                            else:
                                print(f"Step 5a: No record with 'Pending' status found, skipping update and upload.")
                                logging.info(f"Step 5a: No record with 'Pending' status found, skipping update and upload.")
                        else:
                            print(f"Step 5a: No local transcript found for video ID {video_id}")
                            logging.info(f"Step 5a: No local transcript found for video ID {video_id}")

                            # Calling the retrieve_or_generate_transcript method
                            print(f"Step 6: Attempting to retrieve or generate transcript for video ID {video_id}")
                            transcript_text = retrieve_or_generate_transcript(video_id, start_mark, end_mark)

                            if transcript_text:
                                print(f"Step 6: Successfully retrieved or generated transcript for video ID {video_id}")
                                logging.info(f"Step 6: Successfully retrieved or generated transcript for video ID {video_id}")

                                # Save transcript locally
                                with open(full_path_transc, "w", encoding="utf-8") as file:
                                    file.write(transcript_text)
                                logging.info(f"Step 6a: Transcript saved locally at {full_path_transc}.")
                                
                                # Proceed with further steps like uploading to FTP or Google Cloud
                                ftp_upload(full_path_transc, txt_storage_file)
                                upload_blob("spchtotxt1stop", full_path_transc, txt_storage_file)
                                logging.info(f"Transcript uploaded to FTP and Google Cloud.")
                                
                                # Update transcript record in the database
                                update_transcript_record(txt_storage_file, clientes_index_no)
                            else:
                                print(f"Step 6: Failed to retrieve or generate transcript for video ID {video_id}")
                                logging.error(f"Step 6: Failed to retrieve or generate transcript for video ID {video_id}")

                    else:
                        print(f"Step 5a: No YouTube address found for clientes_index_no {clientes_index_no}")
                        logging.info(f"Step 5a: No YouTube address found for clientes_index_no {clientes_index_no}")

                    #-------------------------
                    # Step 10: Generate content using AI (Gemini) if transcript exists
                    if transcript_text:
                        print("Step 10: Transcript exists, proceeding with AI content generation.")
                        logging.info("Step 10: Transcript exists, proceeding with AI content generation.")

                        # Detect language from transcript_text
                        detected_language = detect_full_language(transcript_text)
                        print(f"Step 10: Detected language is {detected_language}")
                        logging.info(f"Step 10: Detected language is {detected_language}")

                        # Get the correct prompt for the resource type and language
                        resource_type = type_info.rstrip()  # Use the type_info as the resource type
                        prompt = select_prompt_tbl(resource_type, detected_language)
                        print(f"Step 10: Retrieved prompt for resource type '{resource_type}' and language '{detected_language}'.")
                        logging.info(f"Step 10: Retrieved prompt for resource type '{resource_type}' and language '{detected_language}'.")

                        if prompt != "ERROR FINDING PROMPT":
                            # Generate content using Gemini AI
                            gemini_output = generate_gemini_content(transcript_text, prompt)

                            if gemini_output:
                                # Save AI content locally and upload to FTP
                                output_filename = f"{clientes_index_no}_{video_id}_{quote_index_no}.txt"
                                output_path = os.path.join(file_path, output_filename)

                                with open(output_path, "w", encoding="utf-8") as output_file:
                                    output_file.write(gemini_output)
                                logging.info(f"AI content saved locally at {output_path}.")

                                # FTP Upload the file to the web server
                                ftp_upload(output_path, output_filename)
                                logging.info(f"AI content uploaded to FTP as {output_filename}.")

                                # Update the gComment field in the Quote table with the filename
                                update_query = """
                                    UPDATE [DB_164462_kdsisdlo291].[dbo].[Quote]
                                    SET [gComment] = ?, [produc] = 'Completed'
                                    WHERE [LockQ] = ? AND [type] = ?
                                """
                                params = (output_filename, clientes_index_no, resource_type)
                                execute_update(update_query, params)
                                logging.info(f"Updated gComment field with {output_filename} for quote_index_no {quote_index_no}.")
                            else:
                                print("Step 10: AI content generation failed.")
                                logging.error("Step 10: AI content generation failed.")
                        else:
                            print("Step 10: No valid prompt found, skipping AI content generation.")
                            logging.error("Step 10: No valid prompt found, skipping AI content generation.")
                    else:
                        print("Step 10: No transcript available, skipping AI content generation.")
                        logging.error("Step 10: No transcript available, skipping AI content generation.")

            else:
                print("Step 11: No pending records found. Waiting for the next interval.")
                logging.info("Step 11: No pending records found. Waiting for the next interval.")

            print(f"Step 12: Sleeping for {interval / 60} minutes.")
            logging.info(f"Step 12: Sleeping for {interval / 60} minutes as of {datetime.now()}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Step 13: Processing loop stopped by the user.")
        logging.info("Step 13: Processing loop stopped by the user.")
    except Exception as e:
        print(f"Step 14: An unexpected error occurred: {e}. Please check the log file for details.")
        logging.error(f"Step 14: An unexpected error occurred: {e}. Please check the log file for details.")

def extract_video_id(url):
    """
    Extracts the video ID from a YouTube URL.
    """
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if video_id_match:
        video_id = video_id_match.group(1)
        logging.info(f"Extracted video ID: {video_id} from URL: {url}")
        return video_id
    logging.warning(f"Failed to extract video ID from URL: {url}")
    return None

# Retrieves YouTube transcript based on time range
def get_youtube_transcript_with_time(video_id, start_time_sec, end_time_sec):
    try:
        # Retrieve transcript with a specified time range
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        full_transcript = " ".join([entry['text'] for entry in transcript if start_time_sec <= entry['start'] <= end_time_sec])
        return full_transcript
    except Exception as e:
        logging.error(f"Error retrieving YouTube transcript with time: {e}")
        return None

# Retrieves transcript from database using index_no_pk
def get_db_transcript(index_no_pk):
    query = "SELECT TOP (1) [indexNoPk],[LockQ],[type],[username],[gComment],[produc],[CustName] FROM [DB_164462_kdsisdlo291].[dbo].[Quote] WHERE [indexNoPk] = ? AND [type] = 'Transcript'"
    return execute_query(query, (index_no_pk,))

# def get_transcript(video_id, start_time_sec, end_time_sec):
#     """
#     Attempts to retrieve YouTube transcript within specified time marks.
#     """
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         filtered_transcript = [entry['text'] for entry in transcript 
#                                if start_time_sec <= entry['start'] <= end_time_sec]
#         full_transcript = " ".join(filtered_transcript)
#         logging.info(f"Retrieved and filtered transcript for video ID {video_id}.")
#         return full_transcript
#     except Exception as e:
#         logging.error(f"Error retrieving transcript from YouTube: {e}")
#         return None

def add_shortcode(shortcode_name, function_name):
    """
    Dummy function to simulate adding a shortcode in WordPress.
    In reality, this should be handled within WordPress.
    """
    # This function is a placeholder. Actual implementation requires WordPress environment.
    pass

if __name__ == "__main__":
    interval_seconds = 30  # Adjust the interval as needed
    logging.info("Script started.")
    run_periodically(interval_seconds)
