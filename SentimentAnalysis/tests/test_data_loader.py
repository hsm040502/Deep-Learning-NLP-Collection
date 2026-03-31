import pytest
from pathlib import Path
from typing import List, Tuple

from src.data_loader import load_reviews_from_xml, ReviewParseError

# Test cases for different XML structures
XML_CONTENT_VALID_STRUCTURE_1 = """
<reviews>
    <review>
        <text>This is a great phone, I love it!</text>
        <sentiment>positive</sentiment>
    </review>
    <review>
        <text>The battery life is terrible.</text>
        <sentiment>negative</sentiment>
    </review>
</reviews>
"""
EXPECTED_REVIEWS_STRUCTURE_1 = [
    ("This is a great phone, I love it!", "positive"),
    ("The battery life is terrible.", "negative"),
]

XML_CONTENT_VALID_STRUCTURE_2_ITEM_TAGS = """
<dataset>
    <item>
        <content>Amazing camera quality.</content>
        <positive/>
    </item>
    <item>
        <content>Screen flickers sometimes.</content>
        <negative/>
    </item>
</dataset>
"""
EXPECTED_REVIEWS_STRUCTURE_2 = [
    ("Amazing camera quality.", "positive"),
    ("Screen flickers sometimes.", "negative"),
]

XML_CONTENT_VALID_STRUCTURE_3_SENTIMENT_ATTRIBUTE = """
<comments>
    <comment sentiment="positive">Great value for money.</comment>
    <comment sentiment="negative">Overheats quickly.</comment>
    <comment sentiment="positive">
        <reviewtext>Good sound.</reviewtext>
    </comment>
</comments>
"""
# Note: The data_loader might pick up "Good sound." as text for the third comment if it finds <reviewtext> first.
# The current data_loader prioritizes specific text tags over the direct content of the review item tag if both exist.
# If the <comment> tag itself contains text AND a <reviewtext> sub-tag, behavior might depend on parsing order.
# For <comment sentiment="positive">Great value for money.</comment>, it should correctly get text and sentiment.
EXPECTED_REVIEWS_STRUCTURE_3 = [
    ("Great value for money.", "positive"),
    ("Overheats quickly.", "negative"),
    ("Good sound.", "positive"), # Assuming <reviewtext> is found
]

XML_CONTENT_NO_REVIEWS = """
<reviews>
    <metadata>Some info</metadata>
</reviews>
"""

XML_CONTENT_MALFORMED = """
<reviews>
    <review>
        <text>This is a great phone, I love it!</text>
        <sentiment>positive
    <!-- Missing closing sentiment tag -->
    </review>
</reviews>
"""

XML_CONTENT_TEXT_ONLY = """
<reviews>
    <review>
        <text>This is a great phone, I love it!</text>
        <!-- No sentiment tag -->
    </review>
    <review>
        <text>The battery life is terrible.</text>
        <sentiment>negative</sentiment>
    </review>
</reviews>
"""
# Expect only the review with a sentiment tag to be loaded
EXPECTED_REVIEWS_TEXT_ONLY_CASE = [
    ("The battery life is terrible.", "negative"),
]

XML_CONTENT_SENTIMENT_ONLY = """
<reviews>
    <review>
        <sentiment>positive</sentiment>
        <!-- No text tag -->
    </review>
    <review>
        <text>The battery life is terrible.</text>
        <sentiment>negative</sentiment>
    </review>
</reviews>
"""
# Expect only the review with both text and sentiment to be loaded
EXPECTED_REVIEWS_SENTIMENT_ONLY_CASE = [
    ("The battery life is terrible.", "negative"),
]

XML_CONTENT_EMPTY_TAGS = """
<reviews>
    <review>
        <text></text>
        <sentiment>positive</sentiment>
    </review>
    <review>
        <text>Non-empty</text>
        <sentiment></sentiment> <!-- Empty sentiment might be ignored -->
    </review>
     <review>
        <text>Good one</text>
        <sentiment>positive</sentiment>
    </review>
</reviews>
"""
EXPECTED_REVIEWS_EMPTY_TAGS = [
    # First review might be skipped if text is empty and loader checks for non-empty strings.
    # Second review might be skipped if sentiment tag is empty.
    # Current loader: empty text content -> skipped; empty sentiment -> no label -> skipped.
    ("Good one", "positive"),
]


@pytest.fixture
def create_xml_file(tmp_path: Path) -> Path:
    """Utility fixture to create a temporary XML file with given content."""
    def _creator(filename: str, content: str) -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content, encoding="utf-8")
        return file_path
    return _creator

def test_load_reviews_valid_structure_1(create_xml_file):
    """Test loading with a common valid XML structure."""
    xml_file = create_xml_file("test1.xml", XML_CONTENT_VALID_STRUCTURE_1)
    reviews = load_reviews_from_xml(xml_file)
    assert reviews == EXPECTED_REVIEWS_STRUCTURE_1

def test_load_reviews_valid_structure_2_item_tags(create_xml_file):
    """Test loading with different valid item and sentiment tags."""
    xml_file = create_xml_file("test2.xml", XML_CONTENT_VALID_STRUCTURE_2_ITEM_TAGS)
    reviews = load_reviews_from_xml(xml_file)
    assert reviews == EXPECTED_REVIEWS_STRUCTURE_2

def test_load_reviews_valid_structure_3_sentiment_attribute(create_xml_file):
    """Test loading with sentiment as an attribute and mixed content."""
    xml_file = create_xml_file("test3.xml", XML_CONTENT_VALID_STRUCTURE_3_SENTIMENT_ATTRIBUTE)
    reviews = load_reviews_from_xml(xml_file)
    # Sort for comparison due to potential find_all order variations if structures are complex
    assert sorted(reviews) == sorted(EXPECTED_REVIEWS_STRUCTURE_3)

def test_load_reviews_file_not_found():
    """Test attempting to load a non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_reviews_from_xml(Path("non_existent_file.xml"))

def test_load_reviews_no_reviews_in_xml(create_xml_file):
    """Test loading an XML file that contains no review items based on parser logic."""
    xml_file = create_xml_file("test_no_reviews.xml", XML_CONTENT_NO_REVIEWS)
    reviews = load_reviews_from_xml(xml_file)
    assert reviews == []

def test_load_reviews_malformed_xml(create_xml_file):
    """Test loading a malformed XML file.
    BeautifulSoup might still parse some parts or raise an error depending on malformation.
    The current load_reviews_from_xml wraps BeautifulSoup parsing in a try-except 
    and raises ReviewParseError.
    """
    xml_file = create_xml_file("test_malformed.xml", XML_CONTENT_MALFORMED)
    # Depending on the severity of malformation, BeautifulSoup might still parse something
    # or the parser might fail. load_reviews_from_xml should catch errors during open/read/parse.
    with pytest.raises(ReviewParseError): # Expecting our custom error
        load_reviews_from_xml(xml_file)

def test_load_reviews_text_only(create_xml_file):
    """Test XML where some reviews only have text but no sentiment tag."""
    xml_file = create_xml_file("test_text_only.xml", XML_CONTENT_TEXT_ONLY)
    reviews = load_reviews_from_xml(xml_file)
    assert reviews == EXPECTED_REVIEWS_TEXT_ONLY_CASE

def test_load_reviews_sentiment_only(create_xml_file):
    """Test XML where some reviews only have sentiment but no text tag/content."""
    xml_file = create_xml_file("test_sentiment_only.xml", XML_CONTENT_SENTIMENT_ONLY)
    reviews = load_reviews_from_xml(xml_file)
    assert reviews == EXPECTED_REVIEWS_SENTIMENT_ONLY_CASE

def test_load_reviews_empty_tags(create_xml_file):
    """Test XML with empty text or sentiment tags."""
    xml_file = create_xml_file("test_empty_tags.xml", XML_CONTENT_EMPTY_TAGS)
    reviews = load_reviews_from_xml(xml_file)
    assert reviews == EXPECTED_REVIEWS_EMPTY_TAGS

def test_load_empty_xml_file(create_xml_file):
    """Test loading an entirely empty XML file."""
    xml_file = create_xml_file("empty.xml", "")
    # BeautifulSoup might parse an empty string as an empty document.
    # Our loader should return an empty list of reviews.
    reviews = load_reviews_from_xml(xml_file)
    assert reviews == [] 