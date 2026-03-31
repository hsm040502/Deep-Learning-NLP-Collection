import logging
from pathlib import Path
from typing import List, Tuple, Optional

from bs4 import BeautifulSoup

# Configure logging
logger = logging.getLogger(__name__)

class ReviewParseError(Exception):
    """Custom exception for errors during review parsing."""
    pass

def load_reviews_from_xml(file_path: Path) -> List[Tuple[str, str]]:
    """Loads reviews and their sentiment labels from an XML file.

    Assumes the XML structure contains items (e.g., <review> or <item> tags)
    each with review text (e.g., within a <text> or <content> tag) and a sentiment
    indicated by the presence of a <positive> or <negative> tag directly
    within the item, or as the content of a <sentiment> tag.

    Args:
        file_path: Path to the XML file.

    Returns:
        A list of tuples, where each tuple contains (review_text, sentiment_label).
        Sentiment labels are 'positive' or 'negative'.

    Raises:
        FileNotFoundError: If the XML file does not exist.
        ReviewParseError: If the XML structure is not as expected or data is missing.
    """
    if not file_path.exists():
        logger.error(f"Data file not found: {file_path}")
        raise FileNotFoundError(f"Data file not found: {file_path}")

    reviews: List[Tuple[str, str]] = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "xml")
    except Exception as e:
        logger.error(f"Error reading or parsing XML file {file_path}: {e}")
        raise ReviewParseError(f"Error reading or parsing XML file {file_path}: {e}")

    # Try to find all review items. Common tags are <review> or <item>.
    # PRD does not specify the exact tag names.
    # We'll try a few common ones or expect a general structure.
    # For Comments_Mobile_Phone.xml, it is often <sentence> or <reviewtext>
    # within a general <review> or <item> tag.
    # Let's assume each comment is within a <sentence> tag based on typical datasets.
    # If these tags are incorrect, this part will need adjustment.

    potential_review_item_tags = ["review", "item", "comment", "entry", "sentence"]
    review_items = []
    for tag_name in potential_review_item_tags:
        review_items = soup.find_all(tag_name)
        if review_items:
            logger.info(f"Found {len(review_items)} items using tag '<{tag_name}>'.")
            break
    
    if not review_items:
        logger.warning(
            f"No review items found with tags {potential_review_item_tags} in {file_path}. "
            f"Please check XML structure or update parser logic."
        )
        return []

    for item_idx, item in enumerate(review_items):
        text_content: Optional[str] = None
        sentiment_label: Optional[str] = None

        # Attempt to find text content
        # Common tags for text: <text>, <content>, <review_text>, <summary>, <description>
        # Or, the text could be the direct string content of the item tag itself if no specific sub-tag.
        potential_text_tags = ["text", "content", "review_text", "reviewtext", "summary", "description"]
        text_tag_found = False
        for text_tag_name in potential_text_tags:
            text_element = item.find(text_tag_name)
            if text_element and text_element.string:
                text_content = text_element.string.strip()
                text_tag_found = True
                break
        
        if not text_tag_found and item.string and item.string.strip(): # Check direct content of review item tag
            # This handles cases like <sentence sentiment="positive">Review text here</sentence>
            # Or <sentence>Review text here <positive/></sentence>
            # If the primary content of the item is the review text itself. 
            if not item.find_all(True, recursive=False): # no child elements, likely direct text
                 text_content = item.string.strip()
            elif item.contents:
                # Collect all navigable strings that are direct children or within simple wrapper tags
                # This is a heuristic to avoid grabbing sentiment tags as part of text.
                current_text_parts = []
                for child_content in item.contents:
                    if isinstance(child_content, str):
                        current_text_parts.append(child_content.strip())
                    elif child_content.name not in ["positive", "negative", "sentiment"]:
                        # If it's a tag but not a sentiment tag, get its string content
                        # This might be too greedy if there are complex nested structures not meant for text
                        if child_content.string:
                            current_text_parts.append(child_content.string.strip())
                text_content = " ".join(filter(None, current_text_parts)).strip()
        
        if not text_content:
             # If still no text, try to get all text within the item, excluding known sentiment tags
            all_strings = []
            for string_part in item.stripped_strings:
                # Check parentage to avoid double-adding from sentiment tags if they also contain text
                parent_is_sentiment = False
                current_parent = string_part.parent
                while current_parent != item and current_parent is not None:
                    if current_parent.name in ["positive", "negative", "sentiment"]:
                        parent_is_sentiment = True
                        break
                    current_parent = current_parent.parent
                if not parent_is_sentiment:
                    all_strings.append(string_part)
            if all_strings:
                text_content = " ".join(all_strings).strip()

        # Attempt to find sentiment label
        # PRD: "XML 中 'positive' 標籤代表正面情感，'negative' 標籤代表負面情感"
        # This could mean <positive/> or <positive>some text</positive> or <sentiment>positive</sentiment>
        if item.find("positive"): # Checks for existence of a <positive> tag
            sentiment_label = "positive"
        elif item.find("negative"): # Checks for existence of a <negative> tag
            sentiment_label = "negative"
        else:
            # Fallback: check a <sentiment> tag's content or an attribute
            sentiment_tag = item.find("sentiment")
            if sentiment_tag and sentiment_tag.string:
                label_text = sentiment_tag.string.strip().lower()
                if "positive" in label_text:
                    sentiment_label = "positive"
                elif "negative" in label_text:
                    sentiment_label = "negative"
            elif item.get("sentiment"): # Check attribute on the item tag itself
                attr_label = item["sentiment"].strip().lower()
                if "positive" in attr_label:
                    sentiment_label = "positive"
                elif "negative" in attr_label:
                    sentiment_label = "negative"

        if text_content and sentiment_label:
            reviews.append((text_content, sentiment_label))
        elif text_content and not sentiment_label:
            logger.warning(
                f"Review item {item_idx+1} has text but no clear sentiment label. Text: '{text_content[:50]}...' Skipping."
            )
        elif not text_content and sentiment_label:
             logger.warning(
                f"Review item {item_idx+1} has a sentiment label ('{sentiment_label}') but no discernible text content. Skipping."
            )
        # If neither, it's probably not a valid review item or parsing failed for it.

    if not reviews:
        logger.warning(
            f"No reviews with both text and sentiment could be extracted from {file_path}. "
            f"Please check XML structure and parsing logic in data_loader.py."
        )
    else:
        logger.info(f"Successfully loaded {len(reviews)} reviews from {file_path}.")

    return reviews

if __name__ == "__main__":
    # This is a simple test/demonstration for the data_loader module.
    # To run this, you would execute `python -m src.data_loader` from the project root.
    logging.basicConfig(level=logging.INFO)
    
    # Create a dummy XML file for testing
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    dummy_xml_path = data_dir / "dummy_comments.xml"

    # Heuristic: Try to use the PRD specified file if it exists, otherwise use dummy.
    # This part is for standalone testing of data_loader.py.
    # In the main application, the actual file path will be passed.
    prd_xml_file = data_dir / "Comments_Mobile_Phone.xml"
    target_xml_file = prd_xml_file if prd_xml_file.exists() else dummy_xml_path

    if target_xml_file == dummy_xml_path and not dummy_xml_path.exists():
        logger.info(f"Creating dummy XML file for testing: {dummy_xml_path}")
        dummy_xml_content = """
        <reviews>
            <review>
                <text>This is a great phone, I love it!</text>
                <sentiment>positive</sentiment>
            </review>
            <item>
                <content>The battery life is terrible.</content>
                <negative/>
            </item>
            <sentence>
                Just okay, nothing special.
                <!-- No explicit positive/negative tag, might be skipped or need neutral handling later -->
            </sentence>
            <comment sentiment="positive">
                Amazing camera quality.
            </comment>
             <review>
                <positive/>
                <reviewtext>This phone is fantastic, and the screen is vibrant.</reviewtext>
            </review>
            <review>
                <reviewtext>Screen flickers sometimes.</reviewtext>
                <sentiment>negative</sentiment>
            </review>
        </reviews>
        """
        with open(dummy_xml_path, "w", encoding="utf-8") as f_dummy:
            f_dummy.write(dummy_xml_content)
    elif target_xml_file == prd_xml_file:
        logger.info(f"Attempting to load PRD specified XML: {prd_xml_file}")
    else:
        logger.info(f"Using existing dummy XML: {dummy_xml_path}")

    try:
        loaded_reviews = load_reviews_from_xml(target_xml_file)
        if loaded_reviews:
            logger.info(f"Successfully loaded {len(loaded_reviews)} reviews:")
            for i, (text, label) in enumerate(loaded_reviews):
                logger.info(f"  Review {i+1}: Label='{label}', Text='{text[:70]}...'" )
        else:
            logger.warning(f"No reviews were loaded from {target_xml_file}. Check file content and parsing logic.")
    except FileNotFoundError:
        logger.error(
            f"Test XML file not found: {target_xml_file}. "
            f"Please place 'Comments_Mobile_Phone.xml' in the '{data_dir}' directory or ensure dummy creation works."
        )
    except ReviewParseError as e:
        logger.error(f"Error parsing review XML: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during data loading test: {e}")

    # Clean up dummy file if it was created by this test run and is the dummy file
    # Be careful not to delete the actual PRD file if it was used for testing.
    if target_xml_file == dummy_xml_path and str(dummy_xml_path.name) == "dummy_comments.xml":
        # A bit of a safeguard: only delete if we definitely know it's the dummy
        # and we intended to use the dummy (i.e., PRD file wasn't found for testing).
        # This logic path is primarily for when the PRD file is missing and dummy is created.
        # If Comments_Mobile_Phone.xml exists, dummy_xml_path won't be target_xml_file unless explicitly set.
        # To be absolutely safe, this cleanup might be better handled manually or with a dedicated test setup/teardown.
        # For now, we'll only delete if it's the default dummy and it exists.
        if dummy_xml_path.exists():
            # logger.info(f"Cleaning up dummy XML file: {dummy_xml_path}")
            # dummy_xml_path.unlink() # Uncomment to enable cleanup
            logger.info(f"Test run finished. If a dummy XML was created, it's at {dummy_xml_path}. Consider manual cleanup if desired.")
            pass # Decided against auto-deletion for now to allow inspection. 