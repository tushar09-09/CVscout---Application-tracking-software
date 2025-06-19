import nltk
import ssl

# This block is to handle potential SSL certificate issues that can prevent NLTK downloads
# on some systems. It's a common workaround.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't have _create_unverified_https_context
    pass
else:
    # Disable SSL certificate verification for NLTK downloads
    ssl._create_default_https_context = _create_unverified_https_context

print("Starting NLTK data downloads...")

# Download individual NLTK data packages
try:
    nltk.download('punkt')
    print(" 'punkt' downloaded or already up-to-date.")
except Exception as e:
    print(f"Error downloading 'punkt': {e}")

try:
    nltk.download('stopwords')
    print(" 'stopwords' downloaded or already up-to-date.")
except Exception as e:
    print(f"Error downloading 'stopwords': {e}")

try:
    nltk.download('wordnet')
    print(" 'wordnet' downloaded or already up-to-date.")
except Exception as e:
    print(f"Error downloading 'wordnet': {e}")

# This is the specific one that was causing the LookupError
try:
    nltk.download('punkt_tab')
    print(" 'punkt_tab' downloaded or already up-to-date.")
except Exception as e:
    print(f"Error downloading 'punkt_tab': {e}")

print("\nNLTK data download process finished. You can now start your Flask app.")

