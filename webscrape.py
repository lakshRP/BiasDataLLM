
# approximetly 1 GB of data (most likely under that)
import requests
from bs4 import BeautifulSoup
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# Function to fetch and parse a Wikipedia article
def fetch_wiki_article(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

# Function to extract and clean the text from paragraphs in a Wikipedia article
def extract_article_text(soup):
    content = soup.find_all('p')
    text = '\n'.join([para.get_text() for para in content if para.get_text(strip=True)])
    
    return text

# Function to find internal Wikipedia links in an article
def find_internal_links(soup):
    links = []
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('/wiki/') and ':' not in href:  # Avoid special pages
            full_url = 'https://en.wikipedia.org' + href
            links.append(full_url)
    return links

# Function to append content to a text file with thread-safety using a lock
def append_to_file(content, file_path, lock, remaining_lines):
    if not content:
        return remaining_lines
    
    lines = content.splitlines()
    lines_to_write = lines[:remaining_lines]  # Limit the number of lines to write to the file

    with lock:  # Ensure only one thread writes to the file at a time
        with open(file_path, 'a', encoding='utf-8') as f:  # Use UTF-8 encoding to avoid issues
            f.write('\n'.join(lines_to_write) + "\n")
    
    return remaining_lines - len(lines_to_write)

# Function to scrape a single article
def scrape_article(url, output_file, lock, remaining_lines):
    if remaining_lines <= 0:
        return None

    soup = fetch_wiki_article(url)
    if soup is None:
        return None

    article_content = extract_article_text(soup)
    remaining_lines = append_to_file(article_content, output_file, lock, remaining_lines)
    
    new_links = find_internal_links(soup)
    return {'url': url, 'links': new_links, 'remaining_lines': remaining_lines}

# Main scraping function using ThreadPoolExecutor for multithreading
def wiki_scraper(seed_urls, output_file, max_lines, max_workers=10):
    visited_urls = set()  # Keep track of visited URLs to avoid repeats
    url_queue = seed_urls.copy()  # Use seed URLs as the starting points
    total_lines_written = 0
    submission_count = 0
    remaining_lines = max_lines
    lock = Lock()  # Create a lock for thread-safe file writing

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        while url_queue and remaining_lines > 0:
            current_url = url_queue.pop(0)
            if current_url in visited_urls:
                continue

            submission_count += 1
            future = executor.submit(scrape_article, current_url, output_file, lock, remaining_lines)
            futures.append(future)

            # Process results as they are completed
            for future in as_completed(futures):
                result = future.result()
                if result:
                    visited_urls.add(result['url'])
                    new_links = result['links']
                    remaining_lines = result['remaining_lines']
                    url_queue.extend(random.sample(new_links, min(len(new_links), 10)))  # Randomly select 10 links

            # Stop once we've written enough lines
            if remaining_lines <= 0:
                break

        print(f"Scraping completed for {output_file}. {max_lines - remaining_lines} lines written.")

# Seed URLs for different fields to create diverse datasets
seed_urls_dict = {
    'technology': ['https://en.wikipedia.org/wiki/Python_(programming_language)', 'https://en.wikipedia.org/wiki/Artificial_intelligence'],
    'history': ['https://en.wikipedia.org/wiki/History_of_medicine', 'https://en.wikipedia.org/wiki/Ancient_Rome'],
    'science': ['https://en.wikipedia.org/wiki/Quantum_mechanics', 'https://en.wikipedia.org/wiki/Evolution'],
    'philosophy': ['https://en.wikipedia.org/wiki/Philosophy', 'https://en.wikipedia.org/wiki/Existentialism'],
    'literature': ['https://en.wikipedia.org/wiki/Shakespeare', 'https://en.wikipedia.org/wiki/Poetry'],
    'economics': ['https://en.wikipedia.org/wiki/Microeconomics', 'https://en.wikipedia.org/wiki/Keynesian_economics'],
    'arts': ['https://en.wikipedia.org/wiki/Painting', 'https://en.wikipedia.org/wiki/Music'],
    'sports': ['https://en.wikipedia.org/wiki/Olympic_Games', 'https://en.wikipedia.org/wiki/Soccer'],
    'geography': ['https://en.wikipedia.org/wiki/Geography', 'https://en.wikipedia.org/wiki/Mountain'],
    'medicine': ['https://en.wikipedia.org/wiki/Medical_research', 'https://en.wikipedia.org/wiki/Public_health']
}

# Maximum number of lines per dataset
max_lines = 1000000

# Run the scraper for each dataset
for field, seed_urls in seed_urls_dict.items():
    output_file = f'{field}_dataset.txt'

    # Clear the file before starting
    open(output_file, 'w').close()

    print(f"Starting to scrape dataset for {field}...")
    
    # Scrape and save the dataset
    wiki_scraper(seed_urls, output_file, max_lines, max_workers=10)

    print(f"{field} dataset saved to {output_file}.")
