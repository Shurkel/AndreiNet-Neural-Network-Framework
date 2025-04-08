import subprocess
import os
import sys

# --- Configuration ---
# Default download directory (can be changed by user input)
DEFAULT_DOWNLOAD_DIR = "Anime_Downloads" 
# Output filename format using yt-dlp's template syntax
# See: https://github.com/yt-dlp/yt-dlp#output-template
# Example: Puts files in "Anime Title/Season 01/S01E01 - Episode Title.mp4"
OUTPUT_TEMPLATE = '%(playlist_title)s/Season %(season_number)s/S%(season_number)02dE%(episode_number)02d - %(title)s.%(ext)s'
# Fallback if playlist/season info isn't available
OUTPUT_TEMPLATE_FALLBACK = '%(title)s.%(ext)s' 
# --- End Configuration ---

def check_yt_dlp():
    """Checks if yt-dlp command exists."""
    try:
        # Use 'yt-dlp --version' which is fast and indicates presence
        subprocess.run(['yt-dlp', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("ERROR: 'yt-dlp' command not found.")
        print("Please install yt-dlp first.")
        print("Installation: pip install -U yt-dlp")
        print("You might also need FFmpeg: https://ffmpeg.org/download.html")
        return False
    


def download_anime(url, download_dir):
    """
    Uses yt-dlp to download video(s) from the given URL.
    """
    if not os.path.exists(download_dir):
        try:
            os.makedirs(download_dir)
            print(f"Created download directory: {download_dir}")
        except OSError as e:
            print(f"Error creating directory {download_dir}: {e}")
            return False

    # Construct the base yt-dlp command
    # -P specifies the download path (yt-dlp handles subdirectories based on -o)
    # -o specifies the output filename template
    # --ignore-errors continues downloading other videos in a playlist if one fails
    # --no-warnings suppresses common warnings (optional, remove if you want more verbosity)
    # --console-title shows progress in the terminal title
    # Add any other yt-dlp options you need here
    command = [
        'yt-dlp',
        '-P', download_dir,
        '-o', OUTPUT_TEMPLATE,
        '--ignore-errors',
        # '--no-warnings', # Uncomment if desired
        '--console-title',
        # Add format selection if needed, e.g., best video+audio merged
        # '-f', 'bv*+ba/b', # Example: Best video + best audio / best overall single file
        url  # The URL to download from
    ]

    print("-" * 30)
    print(f"Attempting to download from: {url}")
    print(f"Saving to: {os.path.join(download_dir, '...')}") # Indicate target general area
    print(f"Using filename template: {OUTPUT_TEMPLATE}")
    print("Executing command:")
    print(" ".join(command)) # Show the command being run (useful for debugging)
    print("-" * 30)
    print("yt-dlp output will follow:")

    try:
        # Run yt-dlp. This will stream output directly to the console.
        # Use stderr=subprocess.STDOUT to merge stderr into stdout stream if preferred
        process = subprocess.run(command, check=False) # check=False lets us handle errors manually

        if process.returncode == 0:
            print("-" * 30)
            print("Download process completed successfully (or yt-dlp finished without fatal errors).")
            return True
        else:
            print("-" * 30)
            print(f"yt-dlp exited with error code: {process.returncode}")
            print("There might have been issues with some or all downloads.")
            print("Check the output above for specific errors.")
            # Try with a simpler fallback template if the complex one failed
            print("\nTrying again with a simpler filename template...")
            fallback_command = [
                'yt-dlp', '-P', download_dir, '-o', OUTPUT_TEMPLATE_FALLBACK,
                '--ignore-errors', '--console-title', url
            ]
            print("Executing command:")
            print(" ".join(fallback_command))
            print("-" * 30)
            process_fallback = subprocess.run(fallback_command, check=False)
            if process_fallback.returncode == 0:
                 print("-" * 30)
                 print("Fallback download process completed successfully.")
                 return True
            else:
                print("-" * 30)
                print(f"Fallback download also failed with error code: {process_fallback.returncode}")
                return False

    except FileNotFoundError:
        # This should be caught by check_yt_dlp, but as a safeguard
        print("ERROR: 'yt-dlp' command not found. Is it installed and in your PATH?")
        return False
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        return False

if __name__ == "__main__":
    print("Anime Downloader using yt-dlp")
    print("==============================")
    print("DISCLAIMER:")
    print(" - Ensure you have the right to download the content.")
    print(" - Respect copyright laws. Support creators via legal means.")
    print(" - Use URLs from unofficial sites at your own risk.")
    print("-" * 30)

    if not check_yt_dlp():
        sys.exit(1) # Exit if yt-dlp is not found

    # Get URL from user
    while True:
        video_url = input("Enter the URL of the anime episode or series page: ")
        if video_url.strip():
            break
        else:
            print("URL cannot be empty.")

    # Get download directory from user
    download_path_input = input(f"Enter download directory (leave empty for default '{DEFAULT_DOWNLOAD_DIR}'): ")
    download_directory = download_path_input.strip() if download_path_input.strip() else DEFAULT_DOWNLOAD_DIR

    # Start download
    download_anime(video_url, download_directory)

    print("\nScript finished.")