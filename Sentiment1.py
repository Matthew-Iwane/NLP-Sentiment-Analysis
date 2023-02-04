import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize

# Load a sample resume
resume = "John Doe is a software engineer with 5 years of experience in developing and maintaining software applications. He has expertise in Python, Java, and C++ programming languages, and has worked on several projects involving data analysis and machine learning. He is a team player and has strong communication skills."

# Tokenize the resume into sentences
sentences = nltk.sent_tokenize(resume)

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Check if the person has more than 3 years of experience as a software engineer
experience = False
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
