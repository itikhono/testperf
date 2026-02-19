import json

def get_combined_output(stdout, stderr):
    combined_output = stdout + stderr
    try:
        json.loads(combined_output) # should not raise an exception
    except Exception as e:
        with open("error_output.txt", "w") as f:
            f.write(f"Error: {e}\nOutput: {combined_output}")
        raise Exception(f"Failed to parse JSON: {e}")
    return combined_output
