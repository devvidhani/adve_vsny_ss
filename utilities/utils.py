from datetime import timedelta

def timestamp_to_seconds_or_string_or_datetime(timestamp_str, output_format="seconds"):
    # timestamp_str is timestamp string (e.g., "01:08:56.838")
    # output_format of type ("seconds", "string", or "datetime")
    # Returns The timestamp in the desired format.

    # Splitting the input string by colon and period to extract hours, minutes, seconds, and microseconds

    parts = timestamp_str.split(":")
    if '.' in parts[-1]:
        second, microsecond = parts[-1].split('.')
        microsecond = int(microsecond)
    else:
        second = parts[-1]
        microsecond = 0
    second = int(second)
    
    if len(parts) == 3:
        hour = int(parts[0])
        minute = int(parts[1])
    elif len(parts) == 2:
        hour = 0
        minute = int(parts[0])
    elif len(parts) == 1:
        hour = 0
        minute = 0
        
    # Convert to timedelta to simplify calculations
    td = timedelta(hours=hour, minutes=minute, seconds=second, microseconds=microsecond*1000)
    
    if output_format == "seconds":
        return td.total_seconds()
    elif output_format == "string":
        return "{:02}:{:02}:{:02}.{:03}".format(td.seconds // 3600, (td.seconds // 60) % 60, td.seconds % 60, td.microseconds // 1000)
    elif output_format == "datetime":
        return td
    else:
        raise ValueError("Invalid output format!")
