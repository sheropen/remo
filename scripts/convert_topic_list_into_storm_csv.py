import csv
import requests
from tqdm import tqdm


# Function to convert topic list to CSV
def convert_topic_list_to_csv(input_file, output_file):
    # First, count total lines for the progress bar
    with open(input_file, "r") as infile:
        total_lines = sum(1 for _ in infile)

    with open(input_file, "r") as infile, open(output_file, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(["topic", "url"])  # Write CSV header

        # Add tqdm progress bar
        for line in tqdm(infile, total=total_lines, desc="Processing topics"):
            topic = line.strip()
            url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"

            # Check if URL exists
            try:
                response = requests.head(url)
                if response.status_code == 200:
                    writer.writerow([topic, url])
                else:
                    print(
                        f"Warning: Invalid URL for topic '{topic}' (Status code: {response.status_code})"
                    )
            except requests.RequestException as e:
                print(f"Error checking URL for topic '{topic}': {str(e)}")


# Usage
convert_topic_list_to_csv(
    "/home/junhao/projects/mog/data/topic_list/wiki_stub_test.txt",
    "/home/junhao/projects/mog/data/topic_list/wiki_stub_test1.csv",
)
