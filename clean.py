def clean_file(filepath):
    with open(filepath, 'rb') as f: # Open in binary read mode
        content = f.read()

    # Remove null bytes
    cleaned_content = content.replace(b'\x00', b'') # b'\x00' is the null byte

    with open(filepath, 'wb') as f: # Open in binary write mode
        f.write(cleaned_content)

# --- Specify the paths to your files here ---
# Make sure these paths are correct relative to where you save this script,
# or use absolute paths.

file_to_clean2 = "multi_tool_agent\.env"     # Or the full path if not in the same directory
file_to_clean1 = 'multi_tool_agent\__init__.py'  # Or the full path if not in the same directory


try:
    clean_file(file_to_clean1)
    print(f"Successfully cleaned: {file_to_clean1}")
    clean_file(file_to_clean2)
    print(f"Successfully cleaned: {file_to_clean2}")
    print("\nFiles processed. Remember to check them and test in the ADK web UI.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure the file paths are correct.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")