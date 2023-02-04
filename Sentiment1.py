import nltk
import pymongo
import PyPDF2
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


# Connect to the MongoDB database
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["resumes_db"]
resumes_collection = db["resumes"]


# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Iterate over each resume in the collection
for resume in resumes_collection.find():
    pdf_file = open(resume["file_path"], "rb")
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)
    resume_text = ""
    for page_num in range(pdf_reader.numPages):
        page = pdf_reader.getPage(page_num)
        resume_text += page.extractText()
    pdf_file.close()

    # Check if the person has more than 3 years of experience as a software engineer
    experience = False
    sentences = nltk.sent_tokenize(resume_text)
    for sentence in sentences:
        if "software engineer" in sentence and "years of experience" in sentence:
            if int(sentence.split()[5]) > 3:
                experience = True

    # Determine the overall sentiment of the resume
    sentiments = [sia.polarity_scores(sentence)['compound'] for sentence in sentences]
    average_sentiment = sum(sentiments) / len(sentiments)

    # Check if the person is a good fit for the company
    if experience and average_sentiment >= 0:
        print("The person is a good fit for the company")
    else:
        print("The person is not a good fit for the company")
